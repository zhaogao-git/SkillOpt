from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from skillopt.worker_bundle import (
    WorkerBundleValidationError,
    build_manifest,
    bundle_edit_stats,
    candidate_hash,
    diff_bundles,
    export_bundle_zip,
    manifest_hash,
    materialize_bundle,
    parse_bundle,
    serialize_bundle,
    validate_bundle_edit_limits,
)


def _source_bundle(root: Path) -> tuple[Path, object]:
    source = root / "source"
    (source / "skills" / "workflow").mkdir(parents=True)
    (source / "strategy").mkdir()
    (source / "AGENTS.md").write_text("# Worker\n\nBe precise.\n", encoding="utf-8")
    (source / "skills" / "workflow" / "SKILL.md").write_text(
        "---\nname: workflow\n---\n\nFollow the workflow.\n",
        encoding="utf-8",
    )
    (source / "strategy" / "answer.md").write_text(
        "# Answer strategy\n\nLead with the answer.\n",
        encoding="utf-8",
    )
    manifest = build_manifest(
        source,
        [
            {"path": "strategy/answer.md", "role": "answer_strategy"},
            {"path": "AGENTS.md", "role": "worker_instructions"},
            {"path": "skills/workflow/SKILL.md", "role": "workflow_skill"},
        ],
    )
    return source, manifest


def _files(source: Path) -> dict[str, str]:
    return {
        "AGENTS.md": (source / "AGENTS.md").read_text(encoding="utf-8"),
        "skills/workflow/SKILL.md": (source / "skills" / "workflow" / "SKILL.md").read_text(encoding="utf-8"),
        "strategy/answer.md": (source / "strategy" / "answer.md").read_text(encoding="utf-8"),
    }


def test_codec_round_trip_and_hashes_are_deterministic(tmp_path: Path) -> None:
    source, manifest = _source_bundle(tmp_path)
    files = _files(source)

    first = serialize_bundle(files, manifest)
    second = serialize_bundle(dict(reversed(list(files.items()))), manifest)
    parsed = parse_bundle(first, expected_manifest=manifest)

    assert first == second
    assert parsed.files == files
    assert parsed.manifest == manifest
    assert parsed.manifest_hash == manifest_hash(manifest)
    assert parsed.candidate_hash == candidate_hash(parsed)
    assert candidate_hash(parse_bundle(second, expected_manifest=manifest)) == parsed.candidate_hash


def test_codec_preserves_non_newline_unicode_separators(tmp_path: Path) -> None:
    source, manifest = _source_bundle(tmp_path)
    files = _files(source)
    files["AGENTS.md"] = "before\x0bafter\u2028still-one-protocol-line\n"

    parsed = parse_bundle(serialize_bundle(files, manifest), expected_manifest=manifest)

    assert parsed.files["AGENTS.md"] == files["AGENTS.md"]


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        (
            "header",
            lambda text: text.replace("skillopt-worker-bundle:v1", "skillopt-worker-bundle:v2", 1),
        ),
        (
            "manifest_hash",
            lambda text: text.replace("manifest-sha256:", "manifest-sha256:" + "0", 1),
        ),
        (
            "marker_path",
            lambda text: text.replace('path="AGENTS.md"', 'path="UNKNOWN.md"', 1),
        ),
        (
            "marker_role",
            lambda text: text.replace('role="worker_instructions"', 'role="other"', 1),
        ),
        (
            "marker_hash",
            lambda text: text.replace("base-sha256=", "base-sha256=0", 1),
        ),
        (
            "missing_file",
            lambda text: text[
                : text.index('<!-- file-begin path="AGENTS.md"')
            ]
            + text[text.index('<!-- file-end path="AGENTS.md" -->') + len('<!-- file-end path="AGENTS.md" -->\n') :],
        ),
        (
            "duplicate_file",
            lambda text: text.replace(
                '<!-- worker-bundle-end -->',
                text[
                    text.index('<!-- file-begin path="AGENTS.md"') :
                    text.index('<!-- file-end path="AGENTS.md" -->') + len('<!-- file-end path="AGENTS.md" -->\n')
                ]
                + '<!-- worker-bundle-end -->',
            ),
        ),
    ],
)
def test_codec_rejects_tampered_headers_markers_hashes_and_sections(
    tmp_path: Path,
    name: str,
    mutate,
) -> None:
    source, manifest = _source_bundle(tmp_path)
    document = serialize_bundle(_files(source), manifest)

    with pytest.raises(WorkerBundleValidationError, match="worker bundle|manifest|marker|file|section|path|role|hash"):
        parse_bundle(mutate(document), expected_manifest=manifest)


def test_codec_rejects_changed_manifest_even_with_recomputed_header(tmp_path: Path) -> None:
    source, manifest = _source_bundle(tmp_path)
    document = serialize_bundle(_files(source), manifest)
    changed = json.loads(json.dumps(manifest.to_dict()))
    changed["files"][0]["role"] = "changed"
    changed_manifest_json = json.dumps(changed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    old_json = manifest.canonical_json()
    changed_document = document.replace(old_json, changed_manifest_json)
    changed_document = changed_document.replace(
        manifest_hash(manifest),
        __import__("hashlib").sha256(changed_manifest_json.encode("utf-8")).hexdigest(),
        1,
    )

    with pytest.raises(WorkerBundleValidationError, match="expected manifest"):
        parse_bundle(changed_document, expected_manifest=manifest)


@pytest.mark.parametrize("path", ["../AGENTS.md", "/absolute.md", "a/../../b.md", "a\\b.md", "", "."])
def test_manifest_rejects_unsafe_paths(tmp_path: Path, path: str) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(WorkerBundleValidationError, match="path"):
        build_manifest(source, [{"path": path, "role": "instructions"}])


@pytest.mark.parametrize(
    "path",
    [
        "dir/a role=b.md",
        "_skillopt/virtual_skill.md",
    ],
)
def test_manifest_rejects_protocol_and_export_metadata_collisions(
    tmp_path: Path,
    path: str,
) -> None:
    source = tmp_path / "source"
    target = source.joinpath(*Path(path).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("content\n", encoding="utf-8")

    with pytest.raises(WorkerBundleValidationError, match="reserved|delimiter"):
        build_manifest(source, [{"path": path, "role": "instructions"}])


def test_manifest_rejects_symlink_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = source / "real.md"
    target.write_text("real\n", encoding="utf-8")
    (source / "linked.md").symlink_to(target)

    with pytest.raises(WorkerBundleValidationError, match="symlink"):
        build_manifest(source, [{"path": "linked.md", "role": "instructions"}])


def test_materialization_is_atomic_and_rejects_symlink_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, manifest = _source_bundle(tmp_path)
    files = _files(source)
    files["AGENTS.md"] = "# Worker\n\nUpdated.\n"
    document = serialize_bundle(files, manifest)
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "sentinel.txt").write_text("keep\n", encoding="utf-8")

    real_replace = os.replace
    failed = False

    def fail_candidate_swap(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        nonlocal failed
        if Path(dst) == output and ".staging-" in Path(src).name and not failed:
            failed = True
            raise OSError("simulated swap failure")
        real_replace(src, dst)

    monkeypatch.setattr("skillopt.worker_bundle.os.replace", fail_candidate_swap)
    with pytest.raises(OSError, match="simulated"):
        materialize_bundle(
            document,
            output,
            expected_manifest=manifest,
            precondition_dir=source,
        )
    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "keep\n"

    monkeypatch.setattr("skillopt.worker_bundle.os.replace", real_replace)
    parsed = materialize_bundle(
        document,
        output,
        expected_manifest=manifest,
        precondition_dir=source,
    )
    assert (output / "AGENTS.md").read_text(encoding="utf-8") == files["AGENTS.md"]
    assert not (output / "sentinel.txt").exists()
    assert parsed.candidate_hash == candidate_hash(parsed)

    symlink_target = tmp_path / "real-output"
    symlink_target.mkdir()
    symlink_output = tmp_path / "linked-output"
    symlink_output.symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(WorkerBundleValidationError, match="symlink"):
        materialize_bundle(
            document,
            symlink_output,
            expected_manifest=manifest,
            precondition_dir=source,
        )


def test_materialization_rejects_precondition_mismatch(tmp_path: Path) -> None:
    source, manifest = _source_bundle(tmp_path)
    document = serialize_bundle(_files(source), manifest)
    (source / "AGENTS.md").write_text("changed after manifest\n", encoding="utf-8")

    with pytest.raises(WorkerBundleValidationError, match="precondition"):
        materialize_bundle(
            document,
            tmp_path / "candidate",
            expected_manifest=manifest,
            precondition_dir=source,
        )


def test_diff_and_zip_exports_are_reproducible(tmp_path: Path) -> None:
    source, manifest = _source_bundle(tmp_path)
    baseline = serialize_bundle(_files(source), manifest)
    changed_files = _files(source)
    changed_files["strategy/answer.md"] += "\nEnd with next steps.\n"
    candidate = serialize_bundle(changed_files, manifest)

    diff = diff_bundles(baseline, candidate, expected_manifest=manifest)
    assert "--- a/strategy/answer.md" in diff
    assert "+++ b/strategy/answer.md" in diff
    assert "+End with next steps." in diff

    first_zip = tmp_path / "first.zip"
    second_zip = tmp_path / "second.zip"
    export_bundle_zip(candidate, first_zip, expected_manifest=manifest, baseline_document=baseline)
    export_bundle_zip(candidate, second_zip, expected_manifest=manifest, baseline_document=baseline)

    assert first_zip.read_bytes() == second_zip.read_bytes()
    with zipfile.ZipFile(first_zip) as archive:
        assert archive.namelist() == [
            "AGENTS.md",
            "skills/workflow/SKILL.md",
            "strategy/answer.md",
            "_skillopt/worker_bundle.diff",
            "_skillopt/worker_bundle_manifest.json",
            "_skillopt/virtual_skill.md",
        ]


def test_bundle_edit_stats_and_limits(tmp_path: Path) -> None:
    source, manifest = _source_bundle(tmp_path)
    baseline = serialize_bundle(_files(source), manifest)
    changed_files = _files(source)
    changed_files["AGENTS.md"] += "Ask one clarifying question.\n"
    changed_files["strategy/answer.md"] += "End with a safe next check.\n"
    candidate = serialize_bundle(changed_files, manifest)

    stats = bundle_edit_stats(baseline, candidate, expected_manifest=manifest)

    assert stats.changed_paths == ("AGENTS.md", "strategy/answer.md")
    assert stats.files_changed == 2
    assert stats.edit_operations == 2
    assert stats.lines_added == 2
    assert stats.lines_removed == 0
    assert stats.approximate_tokens_added > 0
    validate_bundle_edit_limits(
        stats,
        max_files_changed=2,
        max_edit_operations=2,
        max_added_tokens=100,
        max_removed_tokens=0,
    )
    with pytest.raises(WorkerBundleValidationError, match="files_changed"):
        validate_bundle_edit_limits(
            stats,
            max_files_changed=1,
            max_edit_operations=2,
            max_added_tokens=100,
            max_removed_tokens=0,
        )
