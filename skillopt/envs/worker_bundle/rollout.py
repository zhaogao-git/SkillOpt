"""External-runner rollout contract for generic worker bundles."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from skillopt.worker_bundle import (
    WorkerBundleEditStats,
    WorkerBundleManifest,
    bundle_edit_stats,
    diff_bundles,
    export_bundle_zip,
    materialize_bundle,
    serialize_bundle,
    validate_bundle_edit_limits,
)


def _read_baseline_files(
    manifest: WorkerBundleManifest,
    base_dir: Path,
) -> dict[str, str]:
    return {
        entry.path: base_dir.joinpath(*entry.path.split("/")).read_text(encoding="utf-8")
        for entry in manifest.files
    }


def _write_candidate_artifacts(
    *,
    out_root: Path,
    skill_content: str,
    baseline_document: str,
    manifest: WorkerBundleManifest,
    candidate_hash: str,
    manifest_hash: str,
    runner_path: str,
    runner_args: list[str],
    edit_stats: WorkerBundleEditStats,
    edit_limits: dict[str, int],
) -> None:
    (out_root / "virtual_skill.md").write_text(skill_content, encoding="utf-8")
    (out_root / "worker_bundle_manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_root / "worker_bundle.diff").write_text(
        diff_bundles(baseline_document, skill_content, expected_manifest=manifest),
        encoding="utf-8",
    )
    export_bundle_zip(
        skill_content,
        out_root / "candidate_worker_bundle.zip",
        expected_manifest=manifest,
        baseline_document=baseline_document,
    )
    (out_root / "candidate_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_hash": candidate_hash,
                "manifest_hash": manifest_hash,
                "runner": {
                    "path": runner_path,
                    "args": runner_args,
                },
                "edit_stats": edit_stats.to_dict(),
                "edit_limits": edit_limits,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _invoke_runner(
    *,
    item: dict[str, Any],
    task_dir: Path,
    bundle_dir: Path,
    candidate_hash: str,
    manifest_hash: str,
    runner_path: str,
    runner_args: list[str],
    runner_timeout: int,
) -> dict[str, Any]:
    request = {
        "schema_version": 1,
        "task": item,
        "bundle_dir": str(bundle_dir.resolve()),
        "candidate_hash": candidate_hash,
        "manifest_hash": manifest_hash,
    }
    request_path = task_dir / "runner_request.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command = [runner_path, *runner_args, "--request", str(request_path.resolve())]
    proc = subprocess.run(
        command,
        cwd=task_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=runner_timeout,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no error output"
        raise RuntimeError(
            f"worker_bundle runner failed for task {item['id']!r} "
            f"with exit code {proc.returncode}: {detail[:1000]}"
        )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"worker_bundle runner returned invalid JSON for task {item['id']!r}"
        ) from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"worker_bundle runner result must be a JSON object for task {item['id']!r}")

    answer = result.get("answer")
    metadata = result.get("metadata", {})
    if not isinstance(answer, str):
        raise RuntimeError(f"worker_bundle runner result.answer must be a string for task {item['id']!r}")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"worker_bundle runner result.metadata must be an object for task {item['id']!r}")
    try:
        hard = float(result["hard"])
        soft = float(result["soft"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"worker_bundle runner result must contain numeric hard and soft scores for task {item['id']!r}"
        ) from exc
    if not all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in (hard, soft)):
        raise RuntimeError(
            "worker_bundle runner result must contain finite numeric hard and soft scores "
            f"between 0 and 1 for task {item['id']!r}"
        )
    return {
        "answer": answer,
        "hard": hard,
        "soft": soft,
        "metadata": metadata,
    }


def run_batch(
    *,
    items: list[dict],
    skill_content: str,
    out_root: str,
    manifest: WorkerBundleManifest,
    base_dir: str,
    runner_path: str,
    runner_args: list[str],
    runner_timeout: int,
    max_files_changed: int,
    max_edit_operations: int,
    max_added_tokens: int,
    max_removed_tokens: int,
) -> list[dict]:
    """Materialize one candidate and evaluate each task through an external runner."""
    output = Path(out_root)
    output.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_dir)
    candidate_dir = output / "candidate"
    baseline_document = serialize_bundle(_read_baseline_files(manifest, base_path), manifest)
    stats = bundle_edit_stats(
        baseline_document,
        skill_content,
        expected_manifest=manifest,
    )
    limits = {
        "max_files_changed": max_files_changed,
        "max_edit_operations": max_edit_operations,
        "max_added_tokens": max_added_tokens,
        "max_removed_tokens": max_removed_tokens,
    }
    validate_bundle_edit_limits(
        stats,
        max_files_changed=max_files_changed,
        max_edit_operations=max_edit_operations,
        max_added_tokens=max_added_tokens,
        max_removed_tokens=max_removed_tokens,
    )
    parsed = materialize_bundle(
        skill_content,
        candidate_dir,
        expected_manifest=manifest,
        precondition_dir=base_path,
    )
    _write_candidate_artifacts(
        out_root=output,
        skill_content=skill_content,
        baseline_document=baseline_document,
        manifest=manifest,
        candidate_hash=parsed.candidate_hash,
        manifest_hash=parsed.manifest_hash,
        runner_path=runner_path,
        runner_args=runner_args,
        edit_stats=stats,
        edit_limits=limits,
    )

    results: list[dict] = []
    for item in items:
        task_id = str(item["id"])
        task_dir = output / "predictions" / task_id
        if task_dir.is_symlink():
            raise RuntimeError(f"worker_bundle prediction directory must not be a symlink: {task_dir}")
        task_dir.mkdir(parents=True, exist_ok=True)
        runner_result = _invoke_runner(
            item=dict(item),
            task_dir=task_dir,
            bundle_dir=candidate_dir,
            candidate_hash=parsed.candidate_hash,
            manifest_hash=parsed.manifest_hash,
            runner_path=runner_path,
            runner_args=runner_args,
            runner_timeout=runner_timeout,
        )
        prompt = str(item["prompt"])
        answer = runner_result["answer"]
        hard = runner_result["hard"]
        soft = runner_result["soft"]
        metadata = runner_result["metadata"]
        conversation = [
            {
                "role": "system",
                "content": (
                    "A generic external runner evaluated the materialized worker bundle "
                    f"candidate {parsed.candidate_hash}."
                ),
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
            {
                "role": "system",
                "content": json.dumps(
                    {"hard": hard, "soft": soft, "metadata": metadata},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        (task_dir / "conversation.json").write_text(
            json.dumps(conversation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(
            {
                "id": task_id,
                "hard": hard,
                "soft": soft,
                "n_turns": 1,
                "fail_reason": "" if hard > 0 else "external runner hard score was zero",
                "task_type": str(item.get("task_type") or "worker_bundle"),
                "task_description": prompt,
                "predicted_answer": answer,
                "question": prompt,
                "response": answer,
                "runner_metadata": metadata,
                "candidate_hash": parsed.candidate_hash,
                "manifest_hash": parsed.manifest_hash,
            }
        )
    (output / "rollouts.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results
