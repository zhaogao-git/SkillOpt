from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from skillopt.envs.worker_bundle.adapter import WorkerBundleAdapter
from skillopt.envs.worker_bundle.dataloader import WorkerBundleDataLoader
from skillopt.envs.worker_bundle.rollout import _invoke_runner
from skillopt.worker_bundle import build_manifest, serialize_bundle, write_manifest


def _write_split(root: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        (split_dir / "tasks.yaml").write_text(
            yaml.safe_dump(
                {
                    "tasks": [
                        {
                            "id": f"{split}-1",
                            "prompt": f"Handle the {split} request.",
                            "task_type": "support",
                            "contract": {"expected_shape": "concise"},
                        }
                    ]
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _write_sealed_split_without_test(root: Path) -> None:
    for split in ("train", "val"):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        key = "taskId" if split == "train" else "id"
        (split_dir / "tasks.yaml").write_text(
            yaml.safe_dump(
                {
                    key: f"{split}-1",
                    "prompt": f"Handle the {split} request.",
                    "task_type": "support",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _write_bundle(root: Path) -> tuple[Path, Path, str]:
    source = root / "bundle-source"
    (source / "skills" / "workflow").mkdir(parents=True)
    (source / "AGENTS.md").write_text("# Worker\n\nBe useful.\n", encoding="utf-8")
    (source / "skills" / "workflow" / "SKILL.md").write_text("# Workflow\n\nInspect, answer.\n", encoding="utf-8")
    manifest = build_manifest(
        source,
        [
            {"path": "AGENTS.md", "role": "worker_instructions"},
            {"path": "skills/workflow/SKILL.md", "role": "workflow_skill"},
        ],
    )
    manifest_path = root / "worker-bundle-manifest.yaml"
    write_manifest(manifest, manifest_path)
    document = serialize_bundle(
        {
            "AGENTS.md": (source / "AGENTS.md").read_text(encoding="utf-8"),
            "skills/workflow/SKILL.md": (source / "skills" / "workflow" / "SKILL.md").read_text(
                encoding="utf-8"
            ),
        },
        manifest,
    )
    return source, manifest_path, document


def _write_fake_runner(root: Path) -> Path:
    runner = root / "fake_worker_bundle_runner.py"
    runner.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--fixed-flag")
parser.add_argument("--request", required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text(encoding="utf-8"))
bundle_dir = Path(request["bundle_dir"])
assert (bundle_dir / "AGENTS.md").exists()
Path(os.environ["FAKE_RUNNER_LOG"]).write_text(
    json.dumps({"request": request, "fixed_flag": args.fixed_flag}, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps({
    "answer": "Runner answer",
    "hard": 1,
    "soft": 0.75,
    "metadata": {"grader": "fake", "candidate_hash": request["candidate_hash"]},
}))
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return runner


def _write_score_runner(root: Path, hard: str, soft: str) -> Path:
    runner = root / f"score_runner_{hard}_{soft}.py"
    runner.write_text(
        f"""#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.parse_args()
print('{{"answer": "Runner answer", "hard": {hard}, "soft": {soft}, "metadata": {{}}}}')
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return runner


@pytest.mark.parametrize(
    ("hard", "soft"),
    [
        ("NaN", "0.5"),
        ("0.5", "Infinity"),
        ("-0.01", "0.5"),
        ("0.5", "1.01"),
    ],
)
def test_worker_bundle_runner_rejects_non_finite_and_out_of_range_scores(
    tmp_path: Path,
    hard: str,
    soft: str,
) -> None:
    task_dir = tmp_path / "task"
    bundle_dir = tmp_path / "bundle"
    task_dir.mkdir()
    bundle_dir.mkdir()

    with pytest.raises(RuntimeError, match="hard and soft scores"):
        _invoke_runner(
            item={"id": "invalid-score"},
            task_dir=task_dir,
            bundle_dir=bundle_dir,
            candidate_hash="candidate",
            manifest_hash="manifest",
            runner_path=str(_write_score_runner(tmp_path, hard, soft)),
            runner_args=[],
            runner_timeout=10,
        )


def test_worker_bundle_dataloader_loads_pre_split_yaml(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    _write_split(split_dir)
    loader = WorkerBundleDataLoader(
        split_dir=str(split_dir),
        split_mode="split_dir",
        allow_test_split=True,
    )

    loader.setup({"split_mode": "split_dir", "split_dir": str(split_dir), "env": "worker_bundle"})

    assert [item["id"] for item in loader.train_items] == ["train-1"]
    assert loader.val_items[0]["contract"]["expected_shape"] == "concise"
    assert loader.build_eval_batch(0, "test", 42).payload[0]["task_type"] == "support"


def test_worker_bundle_dataloader_normalizes_yaml_dates_for_runner_json(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "splits"
    _write_split(split_dir)
    (split_dir / "train" / "tasks.yaml").write_text(
        "taskId: train-1\n"
        "prompt: Handle the train request.\n"
        "reviewedDate: 2026-07-17\n",
        encoding="utf-8",
    )
    loader = WorkerBundleDataLoader(
        split_dir=str(split_dir),
        split_mode="split_dir",
    )

    loader.setup({"split_mode": "split_dir", "split_dir": str(split_dir), "env": "worker_bundle"})

    assert loader.train_items[0]["reviewedDate"] == "2026-07-17"
    assert loader.train_items[0]["_split"] == "train"
    assert loader.val_items[0]["_split"] == "val"
    json.dumps(loader.train_items[0])


def test_worker_bundle_dataloader_accepts_task_id_and_does_not_open_sealed_test(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "splits"
    _write_sealed_split_without_test(split_dir)
    loader = WorkerBundleDataLoader(split_dir=str(split_dir), split_mode="split_dir")

    loader.setup({"split_mode": "split_dir", "split_dir": str(split_dir), "env": "worker_bundle"})

    assert loader.train_items[0]["id"] == "train-1"
    assert loader.train_items[0]["taskId"] == "train-1"
    assert loader.val_items[0]["id"] == "val-1"
    assert loader.test_items == []
    with pytest.raises(PermissionError, match="sealed test"):
        loader.build_eval_batch(0, "test", 42)


def test_worker_bundle_dataloader_rejects_cross_split_duplicate_ids(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    _write_sealed_split_without_test(split_dir)
    (split_dir / "val" / "tasks.yaml").write_text(
        "taskId: train-1\nprompt: duplicate across splits\n",
        encoding="utf-8",
    )
    loader = WorkerBundleDataLoader(split_dir=str(split_dir), split_mode="split_dir")

    with pytest.raises(ValueError, match="across worker_bundle splits"):
        loader.setup({"split_mode": "split_dir", "split_dir": str(split_dir), "env": "worker_bundle"})


def test_worker_bundle_dataloader_lazily_loads_authorized_test_split(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    _write_split(split_dir)
    loader = WorkerBundleDataLoader(
        split_dir=str(split_dir),
        split_mode="split_dir",
        allow_test_split=True,
    )
    loader.setup({"split_mode": "split_dir", "split_dir": str(split_dir), "env": "worker_bundle"})

    assert loader.test_items == []
    test_batch = loader.build_eval_batch(0, "test", 42)
    assert [item["id"] for item in test_batch.payload] == ["test-1"]


def test_worker_bundle_dataloader_rejects_invalid_test_authorization_and_split(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "splits"
    _write_sealed_split_without_test(split_dir)
    with pytest.raises(ValueError, match="boolean"):
        WorkerBundleDataLoader(
            split_dir=str(split_dir),
            allow_test_split="sometimes",
        )

    loader = WorkerBundleDataLoader(split_dir=str(split_dir))
    loader.setup({"split_dir": str(split_dir), "split_mode": "split_dir"})
    with pytest.raises(ValueError, match="Unknown worker_bundle split"):
        loader.build_eval_batch(0, "mystery", 42)


def test_worker_bundle_dataloader_rejects_duplicate_ids(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    _write_split(split_dir)
    duplicate_file = split_dir / "train" / "more.yml"
    duplicate_file.write_text("- id: train-1\n  prompt: duplicate\n", encoding="utf-8")
    loader = WorkerBundleDataLoader(split_dir=str(split_dir), split_mode="split_dir")

    with pytest.raises(ValueError, match="Duplicate worker_bundle task id"):
        loader.setup({"split_mode": "split_dir", "split_dir": str(split_dir), "env": "worker_bundle"})


def test_worker_bundle_dataloader_rejects_unsafe_task_ids(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    _write_split(split_dir)
    (split_dir / "train" / "tasks.yaml").write_text(
        "- id: ../../escape\n  prompt: unsafe\n",
        encoding="utf-8",
    )
    loader = WorkerBundleDataLoader(split_dir=str(split_dir), split_mode="split_dir")

    with pytest.raises(ValueError, match="filesystem-safe"):
        loader.setup({"split_mode": "split_dir", "split_dir": str(split_dir), "env": "worker_bundle"})


def test_worker_bundle_adapter_materializes_and_invokes_external_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_dir = tmp_path / "splits"
    _write_split(split_dir)
    source, manifest_path, document = _write_bundle(tmp_path)
    runner = _write_fake_runner(tmp_path)
    log_path = tmp_path / "runner-log.json"
    monkeypatch.setenv("WORKER_BUNDLE_RUNNER", str(runner))
    monkeypatch.setenv("FAKE_RUNNER_LOG", str(log_path))
    adapter = WorkerBundleAdapter(
        split_dir=str(split_dir),
        split_mode="split_dir",
        bundle_manifest=str(manifest_path),
        bundle_base_dir=str(source),
        runner_env_var="WORKER_BUNDLE_RUNNER",
        runner_args=["--fixed-flag", "fixed-value"],
        runner_timeout=10,
    )
    adapter.setup({"split_mode": "split_dir", "split_dir": str(split_dir), "env": "worker_bundle"})
    items = adapter.build_eval_env(1, "valid_seen", 42)
    out_dir = tmp_path / "rollout"

    results = adapter.rollout(items, document, str(out_dir))

    assert results == [
        {
            "id": "val-1",
            "hard": 1.0,
            "soft": 0.75,
            "n_turns": 1,
            "fail_reason": "",
            "task_type": "support",
            "task_description": "Handle the val request.",
            "predicted_answer": "Runner answer",
            "question": "Handle the val request.",
            "response": "Runner answer",
            "runner_metadata": {
                "grader": "fake",
                "candidate_hash": results[0]["candidate_hash"],
            },
            "candidate_hash": results[0]["candidate_hash"],
            "manifest_hash": results[0]["manifest_hash"],
        }
    ]
    runner_log = json.loads(log_path.read_text(encoding="utf-8"))
    assert runner_log["fixed_flag"] == "fixed-value"
    assert runner_log["request"]["task"]["id"] == "val-1"
    assert Path(runner_log["request"]["bundle_dir"]).is_dir()
    conversation_path = out_dir / "predictions" / "val-1" / "conversation.json"
    conversation = json.loads(conversation_path.read_text(encoding="utf-8"))
    assert [message["role"] for message in conversation] == ["system", "user", "assistant", "system"]
    assert conversation[2]["content"] == "Runner answer"
    assert (out_dir / "candidate" / "AGENTS.md").read_text(encoding="utf-8") == "# Worker\n\nBe useful.\n"

    adapter.export_best_bundle(document, str(out_dir))
    assert (out_dir / "best_worker_bundle.zip").is_file()
    assert (out_dir / "best_worker_bundle" / "AGENTS.md").read_text(encoding="utf-8") == "# Worker\n\nBe useful.\n"
    best_metadata = json.loads(
        (out_dir / "best_worker_bundle_metadata.json").read_text(encoding="utf-8")
    )
    assert best_metadata["candidate_hash"] == results[0]["candidate_hash"]
    assert best_metadata["edit_stats"]["files_changed"] == 0


def test_worker_bundle_adapter_requires_runner_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split_dir = tmp_path / "splits"
    _write_split(split_dir)
    source, manifest_path, _document = _write_bundle(tmp_path)
    monkeypatch.delenv("WORKER_BUNDLE_RUNNER", raising=False)
    adapter = WorkerBundleAdapter(
        split_dir=str(split_dir),
        split_mode="split_dir",
        bundle_manifest=str(manifest_path),
        bundle_base_dir=str(source),
        runner_env_var="WORKER_BUNDLE_RUNNER",
    )

    with pytest.raises(ValueError, match="runner"):
        adapter.setup({"split_mode": "split_dir", "split_dir": str(split_dir), "env": "worker_bundle"})


def test_worker_bundle_adapter_rejects_candidate_over_edit_limits_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_dir = tmp_path / "splits"
    _write_split(split_dir)
    source, manifest_path, document = _write_bundle(tmp_path)
    runner = _write_fake_runner(tmp_path)
    log_path = tmp_path / "runner-log.json"
    monkeypatch.setenv("FAKE_RUNNER_LOG", str(log_path))
    monkeypatch.setenv("FAKE_COPILOT_COUNT", str(tmp_path / "unused-count"))
    adapter = WorkerBundleAdapter(
        split_dir=str(split_dir),
        bundle_manifest=str(manifest_path),
        bundle_base_dir=str(source),
        runner_path=str(runner),
        max_files_changed=0,
    )
    adapter.setup({"split_dir": str(split_dir), "env": "worker_bundle"})
    changed = document.replace("# Worker", "# Changed worker", 1)

    with pytest.raises(ValueError, match="files_changed"):
        adapter.rollout(
            adapter.build_eval_env(1, "valid_seen", 42),
            changed,
            str(tmp_path / "rollout"),
        )
    assert not log_path.exists()


def test_worker_bundle_adapter_rejects_unsupported_mutation_modes(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    _write_split(split_dir)
    source, manifest_path, _document = _write_bundle(tmp_path)
    adapter = WorkerBundleAdapter(
        split_dir=str(split_dir),
        split_mode="split_dir",
        bundle_manifest=str(manifest_path),
        bundle_base_dir=str(source),
        runner_path="python",
    )

    with pytest.raises(ValueError, match="skill_update_mode"):
        adapter.setup(
            {
                "split_mode": "split_dir",
                "split_dir": str(split_dir),
                "env": "worker_bundle",
                "skill_update_mode": "full_rewrite_minibatch",
            }
        )


def test_worker_bundle_is_registered_and_template_flattens() -> None:
    from scripts import eval_only, train

    train._ENV_REGISTRY.clear()
    train._register_builtins()
    eval_only._ENV_REGISTRY.clear()
    eval_only._register_builtins()
    assert train._ENV_REGISTRY["worker_bundle"] is WorkerBundleAdapter
    assert eval_only._ENV_REGISTRY["worker_bundle"] is WorkerBundleAdapter

    cfg = train.load_config(
        Namespace(
            config="configs/worker_bundle/template.yaml",
            cfg_options=[],
            backend=None,
        )
    )
    assert cfg["env"] == "worker_bundle"
    assert cfg["optimizer_backend"] == "copilot_cli"
    assert cfg["target_backend"] == "copilot_cli"
    assert cfg["runner_env_var"] == "WORKER_BUNDLE_RUNNER"
    assert cfg["runner_args"] == []
    assert cfg["allow_test_split"] is False
    assert cfg["max_files_changed"] == 3
    assert cfg["max_edit_operations"] == 8
