"""Canonical virtual multi-file documents for worker instruction bundles."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml

_HEADER = "<!-- skillopt-worker-bundle:v1 -->"
_MANIFEST_BEGIN = "<!-- manifest-json-begin -->"
_MANIFEST_END = "<!-- manifest-json-end -->"
_BUNDLE_END = "<!-- worker-bundle-end -->"
_EXPORT_METADATA_PREFIX = "_skillopt"
_RESERVED_PREFIXES = (
    "<!-- skillopt-worker-bundle:",
    "<!-- manifest-",
    "<!-- file-begin ",
    "<!-- file-end ",
    "<!-- worker-bundle-end ",
    "<!-- worker-bundle-end -->",
)


class WorkerBundleValidationError(ValueError):
    """Raised when a virtual worker bundle violates its immutable contract."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_content(content: str) -> str:
    normalized = str(content).replace("\r\n", "\n").replace("\r", "\n")
    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    for line in normalized.splitlines():
        if line.startswith(_RESERVED_PREFIXES):
            raise WorkerBundleValidationError(
                "Worker bundle file content contains a reserved marker line"
            )
    return normalized


def _validate_path(raw_path: str) -> str:
    path = str(raw_path)
    if not path or path in {".", ".."}:
        raise WorkerBundleValidationError(f"Invalid worker bundle path: {path!r}")
    if "\\" in path or any(ord(character) < 32 for character in path):
        raise WorkerBundleValidationError(f"Invalid worker bundle path: {path!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise WorkerBundleValidationError(f"Unsafe worker bundle path: {path!r}")
    if pure.parts[0] == _EXPORT_METADATA_PREFIX:
        raise WorkerBundleValidationError(
            f"Worker bundle path uses reserved {_EXPORT_METADATA_PREFIX!r} namespace: {path!r}"
        )
    if " role=" in path or " -->" in path:
        raise WorkerBundleValidationError(
            f"Worker bundle path contains a reserved marker delimiter: {path!r}"
        )
    return pure.as_posix()


def _validate_role(raw_role: str, path: str) -> str:
    role = str(raw_role).strip()
    if not role:
        raise WorkerBundleValidationError(f"Worker bundle role is required for path {path!r}")
    if any(ord(character) < 32 for character in role) or " base-sha256=" in role or " -->" in role:
        raise WorkerBundleValidationError(
            f"Worker bundle role contains a reserved marker delimiter for path {path!r}"
        )
    return role


def _check_no_symlink_components(root: Path, relative_path: str | None = None) -> None:
    current = root
    probe = current
    while True:
        if probe.is_symlink():
            raise WorkerBundleValidationError(f"Worker bundle path is a symlink: {probe}")
        if probe == probe.parent:
            break
        probe = probe.parent
    if relative_path is None:
        return
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise WorkerBundleValidationError(f"Worker bundle path is a symlink: {current}")


@dataclass(frozen=True, slots=True)
class WorkerBundleFile:
    path: str
    role: str
    base_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "role": self.role,
            "base_sha256": self.base_sha256,
        }


@dataclass(frozen=True, slots=True)
class WorkerBundleManifest:
    files: tuple[WorkerBundleFile, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "files": [entry.to_dict() for entry in self.files],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerBundleManifest:
        if value.get("schema_version") != 1:
            raise WorkerBundleValidationError(
                f"Unsupported worker bundle schema_version: {value.get('schema_version')!r}"
            )
        raw_files = value.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise WorkerBundleValidationError("Worker bundle manifest must contain a non-empty files list")

        entries: list[WorkerBundleFile] = []
        seen: set[str] = set()
        for raw in raw_files:
            if not isinstance(raw, dict):
                raise WorkerBundleValidationError("Worker bundle manifest file entries must be mappings")
            path = _validate_path(str(raw.get("path", "")))
            role = _validate_role(str(raw.get("role", "")), path)
            base_sha256 = str(raw.get("base_sha256", "")).strip().lower()
            if len(base_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in base_sha256):
                raise WorkerBundleValidationError(f"Invalid base hash for worker bundle path {path!r}")
            if path in seen:
                raise WorkerBundleValidationError(f"Duplicate worker bundle manifest path: {path}")
            seen.add(path)
            entries.append(WorkerBundleFile(path=path, role=role, base_sha256=base_sha256))

        ordered = tuple(sorted(entries, key=lambda entry: entry.path))
        return cls(files=ordered)


@dataclass(frozen=True, slots=True)
class ParsedWorkerBundle:
    manifest: WorkerBundleManifest
    files: dict[str, str]
    manifest_hash: str
    candidate_hash: str


@dataclass(frozen=True, slots=True)
class WorkerBundleEditStats:
    changed_paths: tuple[str, ...]
    edit_operations: int
    lines_added: int
    lines_removed: int
    approximate_tokens_added: int
    approximate_tokens_removed: int

    @property
    def files_changed(self) -> int:
        return len(self.changed_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_paths": list(self.changed_paths),
            "files_changed": self.files_changed,
            "edit_operations": self.edit_operations,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "approximate_tokens_added": self.approximate_tokens_added,
            "approximate_tokens_removed": self.approximate_tokens_removed,
        }


def manifest_hash(manifest: WorkerBundleManifest) -> str:
    return _sha256_bytes(manifest.canonical_json().encode("utf-8"))


def _candidate_hash(manifest_sha256: str, files: Mapping[str, str]) -> str:
    payload = {
        "manifest_sha256": manifest_sha256,
        "files": [
            {
                "path": path,
                "content_sha256": _sha256_bytes(_canonical_content(files[path]).encode("utf-8")),
            }
            for path in sorted(files)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(encoded.encode("utf-8"))


def candidate_hash(bundle: ParsedWorkerBundle) -> str:
    return _candidate_hash(bundle.manifest_hash, bundle.files)


def build_manifest(
    source_dir: str | os.PathLike[str],
    file_specs: Sequence[Mapping[str, str]],
) -> WorkerBundleManifest:
    root = Path(source_dir)
    _check_no_symlink_components(root)
    if not root.is_dir():
        raise WorkerBundleValidationError(f"Worker bundle source directory does not exist: {root}")

    entries: list[WorkerBundleFile] = []
    seen: set[str] = set()
    for spec in file_specs:
        path = _validate_path(str(spec.get("path", "")))
        role = _validate_role(str(spec.get("role", "")), path)
        if path in seen:
            raise WorkerBundleValidationError(f"Duplicate worker bundle manifest path: {path}")
        seen.add(path)
        _check_no_symlink_components(root, path)
        file_path = root.joinpath(*PurePosixPath(path).parts)
        if not file_path.is_file():
            raise WorkerBundleValidationError(f"Worker bundle source file is missing: {path}")
        content = _canonical_content(file_path.read_text(encoding="utf-8"))
        entries.append(
            WorkerBundleFile(
                path=path,
                role=role,
                base_sha256=_sha256_bytes(content.encode("utf-8")),
            )
        )
    return WorkerBundleManifest(files=tuple(sorted(entries, key=lambda entry: entry.path)))


def write_manifest(
    manifest: WorkerBundleManifest,
    path: str | os.PathLike[str],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(manifest.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_manifest(path: str | os.PathLike[str]) -> WorkerBundleManifest:
    manifest_path = Path(path)
    _check_no_symlink_components(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise WorkerBundleValidationError("Worker bundle manifest must be a YAML or JSON mapping")
    return WorkerBundleManifest.from_dict(value)


def serialize_bundle(
    files: Mapping[str, str],
    manifest: WorkerBundleManifest,
) -> str:
    expected_paths = [entry.path for entry in manifest.files]
    actual_paths = sorted(_validate_path(path) for path in files)
    if actual_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(actual_paths))
        unknown = sorted(set(actual_paths) - set(expected_paths))
        raise WorkerBundleValidationError(
            f"Worker bundle file set does not match manifest; missing={missing}, unknown={unknown}"
        )

    lines = [
        _HEADER,
        f"<!-- manifest-sha256:{manifest_hash(manifest)} -->",
        _MANIFEST_BEGIN,
        manifest.canonical_json(),
        _MANIFEST_END,
    ]
    for entry in manifest.files:
        path_json = json.dumps(entry.path, ensure_ascii=False)
        role_json = json.dumps(entry.role, ensure_ascii=False)
        lines.append(
            f"<!-- file-begin path={path_json} role={role_json} base-sha256={entry.base_sha256} -->"
        )
        content = _canonical_content(files[entry.path])
        if content:
            lines.extend(content[:-1].split("\n"))
        lines.append(f"<!-- file-end path={path_json} -->")
    lines.append(_BUNDLE_END)
    return "\n".join(lines) + "\n"


def _parse_json_string(value: str, label: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise WorkerBundleValidationError(f"Invalid worker bundle {label} marker") from exc
    if not isinstance(parsed, str):
        raise WorkerBundleValidationError(f"Invalid worker bundle {label} marker")
    return parsed


def parse_bundle(
    document: str,
    *,
    expected_manifest: WorkerBundleManifest | None = None,
) -> ParsedWorkerBundle:
    normalized_document = str(document).replace("\r\n", "\n").replace("\r", "\n")
    if normalized_document.endswith("\n"):
        normalized_document = normalized_document[:-1]
    lines = normalized_document.split("\n")
    if len(lines) < 6 or lines[0] != _HEADER:
        raise WorkerBundleValidationError("Invalid worker bundle header")
    hash_line = lines[1]
    prefix = "<!-- manifest-sha256:"
    suffix = " -->"
    if not hash_line.startswith(prefix) or not hash_line.endswith(suffix):
        raise WorkerBundleValidationError("Invalid worker bundle manifest hash marker")
    declared_manifest_hash = hash_line[len(prefix) : -len(suffix)]
    if len(declared_manifest_hash) != 64:
        raise WorkerBundleValidationError("Invalid worker bundle manifest hash marker")
    if lines[2] != _MANIFEST_BEGIN:
        raise WorkerBundleValidationError("Missing worker bundle manifest begin marker")
    try:
        manifest_end_index = lines.index(_MANIFEST_END, 3)
    except ValueError as exc:
        raise WorkerBundleValidationError("Missing worker bundle manifest end marker") from exc
    manifest_text = "\n".join(lines[3:manifest_end_index])
    try:
        manifest_value = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise WorkerBundleValidationError("Invalid worker bundle manifest JSON") from exc
    if not isinstance(manifest_value, dict):
        raise WorkerBundleValidationError("Worker bundle manifest JSON must be an object")
    manifest = WorkerBundleManifest.from_dict(manifest_value)
    actual_manifest_hash = manifest_hash(manifest)
    if declared_manifest_hash != actual_manifest_hash:
        raise WorkerBundleValidationError("Worker bundle manifest hash mismatch")
    if expected_manifest is not None and manifest != expected_manifest:
        raise WorkerBundleValidationError("Worker bundle does not match the expected manifest")

    entry_by_path = {entry.path: entry for entry in manifest.files}
    files: dict[str, str] = {}
    index = manifest_end_index + 1
    while index < len(lines):
        line = lines[index]
        if line == _BUNDLE_END:
            index += 1
            break
        marker_prefix = "<!-- file-begin path="
        if not line.startswith(marker_prefix) or not line.endswith(" -->"):
            raise WorkerBundleValidationError(f"Invalid worker bundle file marker at line {index + 1}")
        marker_body = line[len(marker_prefix) : -4]
        try:
            path_part, remainder = marker_body.split(" role=", 1)
            role_part, base_sha256 = remainder.rsplit(" base-sha256=", 1)
        except ValueError as exc:
            raise WorkerBundleValidationError("Invalid worker bundle file begin marker") from exc
        path = _validate_path(_parse_json_string(path_part, "path"))
        role = _parse_json_string(role_part, "role")
        if path in files:
            raise WorkerBundleValidationError(f"Duplicate worker bundle file section: {path}")
        entry = entry_by_path.get(path)
        if entry is None:
            raise WorkerBundleValidationError(f"Unknown worker bundle file path: {path}")
        if role != entry.role:
            raise WorkerBundleValidationError(f"Worker bundle role marker mismatch for {path}")
        if base_sha256 != entry.base_sha256:
            raise WorkerBundleValidationError(f"Worker bundle base hash marker mismatch for {path}")

        end_marker = f"<!-- file-end path={json.dumps(path, ensure_ascii=False)} -->"
        content_lines: list[str] = []
        index += 1
        while index < len(lines) and lines[index] != end_marker:
            if lines[index].startswith("<!-- file-begin "):
                raise WorkerBundleValidationError(f"Missing worker bundle file end marker for {path}")
            content_lines.append(lines[index])
            index += 1
        if index >= len(lines):
            raise WorkerBundleValidationError(f"Missing worker bundle file end marker for {path}")
        content = "\n".join(content_lines)
        if content_lines:
            content += "\n"
        files[path] = _canonical_content(content)
        index += 1
    else:
        raise WorkerBundleValidationError("Missing worker bundle end marker")

    if index != len(lines):
        raise WorkerBundleValidationError("Unexpected content after worker bundle end marker")
    expected_paths = set(entry_by_path)
    actual_paths = set(files)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unknown = sorted(actual_paths - expected_paths)
        raise WorkerBundleValidationError(
            f"Worker bundle file sections do not match manifest; missing={missing}, unknown={unknown}"
        )

    parsed_hash = _candidate_hash(actual_manifest_hash, files)
    return ParsedWorkerBundle(
        manifest=manifest,
        files=files,
        manifest_hash=actual_manifest_hash,
        candidate_hash=parsed_hash,
    )


def _verify_preconditions(manifest: WorkerBundleManifest, precondition_dir: Path) -> None:
    _check_no_symlink_components(precondition_dir)
    if not precondition_dir.is_dir():
        raise WorkerBundleValidationError(
            f"Worker bundle precondition directory does not exist: {precondition_dir}"
        )
    for entry in manifest.files:
        _check_no_symlink_components(precondition_dir, entry.path)
        path = precondition_dir.joinpath(*PurePosixPath(entry.path).parts)
        if not path.is_file():
            raise WorkerBundleValidationError(f"Worker bundle precondition file is missing: {entry.path}")
        content = _canonical_content(path.read_text(encoding="utf-8"))
        if _sha256_bytes(content.encode("utf-8")) != entry.base_sha256:
            raise WorkerBundleValidationError(
                f"Worker bundle precondition hash mismatch for {entry.path}"
            )


def _check_output_path(output_dir: Path) -> None:
    current = output_dir
    while not current.exists() and current != current.parent:
        current = current.parent
    if current.is_symlink():
        raise WorkerBundleValidationError(f"Worker bundle output path contains a symlink: {current}")
    probe = current
    while probe != probe.parent:
        if probe.is_symlink():
            raise WorkerBundleValidationError(f"Worker bundle output path contains a symlink: {probe}")
        probe = probe.parent
    if output_dir.exists():
        if output_dir.is_symlink():
            raise WorkerBundleValidationError(f"Worker bundle output directory is a symlink: {output_dir}")
        if not output_dir.is_dir():
            raise WorkerBundleValidationError(f"Worker bundle output path is not a directory: {output_dir}")


def materialize_bundle(
    document: str,
    output_dir: str | os.PathLike[str],
    *,
    expected_manifest: WorkerBundleManifest | None = None,
    precondition_dir: str | os.PathLike[str] | None = None,
) -> ParsedWorkerBundle:
    parsed = parse_bundle(document, expected_manifest=expected_manifest)
    if precondition_dir is not None:
        _verify_preconditions(parsed.manifest, Path(precondition_dir))

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    _check_output_path(output)
    suffix = uuid.uuid4().hex
    staging = output.parent / f".{output.name}.staging-{suffix}"
    backup = output.parent / f".{output.name}.backup-{suffix}"
    staging.mkdir()
    moved_existing = False
    try:
        for path, content in parsed.files.items():
            destination = staging.joinpath(*PurePosixPath(path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")

        if output.exists():
            os.replace(output, backup)
            moved_existing = True
        try:
            os.replace(staging, output)
        except Exception:
            if moved_existing and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and output.exists():
            shutil.rmtree(backup)
    return parsed


def diff_bundles(
    baseline_document: str,
    candidate_document: str,
    *,
    expected_manifest: WorkerBundleManifest | None = None,
) -> str:
    baseline = parse_bundle(baseline_document, expected_manifest=expected_manifest)
    candidate = parse_bundle(candidate_document, expected_manifest=baseline.manifest)
    chunks: list[str] = []
    for entry in baseline.manifest.files:
        before = baseline.files[entry.path].splitlines(keepends=True)
        after = candidate.files[entry.path].splitlines(keepends=True)
        if before == after:
            continue
        chunks.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{entry.path}",
                tofile=f"b/{entry.path}",
            )
        )
    return "".join(chunks)


def _approximate_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def bundle_edit_stats(
    baseline_document: str,
    candidate_document: str,
    *,
    expected_manifest: WorkerBundleManifest | None = None,
) -> WorkerBundleEditStats:
    baseline = parse_bundle(baseline_document, expected_manifest=expected_manifest)
    candidate = parse_bundle(candidate_document, expected_manifest=baseline.manifest)
    changed_paths: list[str] = []
    operations = lines_added = lines_removed = 0
    tokens_added = tokens_removed = 0
    for entry in baseline.manifest.files:
        before = baseline.files[entry.path].splitlines(keepends=True)
        after = candidate.files[entry.path].splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
        file_changed = False
        for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
            if tag == "equal":
                continue
            file_changed = True
            operations += 1
            removed = "".join(before[before_start:before_end])
            added = "".join(after[after_start:after_end])
            lines_removed += before_end - before_start
            lines_added += after_end - after_start
            tokens_removed += _approximate_token_count(removed)
            tokens_added += _approximate_token_count(added)
        if file_changed:
            changed_paths.append(entry.path)
    return WorkerBundleEditStats(
        changed_paths=tuple(changed_paths),
        edit_operations=operations,
        lines_added=lines_added,
        lines_removed=lines_removed,
        approximate_tokens_added=tokens_added,
        approximate_tokens_removed=tokens_removed,
    )


def validate_bundle_edit_limits(
    stats: WorkerBundleEditStats,
    *,
    max_files_changed: int,
    max_edit_operations: int,
    max_added_tokens: int,
    max_removed_tokens: int,
) -> None:
    limits = {
        "files_changed": (stats.files_changed, max_files_changed),
        "edit_operations": (stats.edit_operations, max_edit_operations),
        "approximate_tokens_added": (stats.approximate_tokens_added, max_added_tokens),
        "approximate_tokens_removed": (stats.approximate_tokens_removed, max_removed_tokens),
    }
    for label, (actual, limit) in limits.items():
        if limit < 0:
            raise ValueError(f"{label} limit must be non-negative")
        if actual > limit:
            raise WorkerBundleValidationError(
                f"Worker bundle edit limit exceeded for {label}: actual={actual}, limit={limit}"
            )


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    return info


def export_bundle_zip(
    document: str,
    output_path: str | os.PathLike[str],
    *,
    expected_manifest: WorkerBundleManifest | None = None,
    baseline_document: str | None = None,
) -> ParsedWorkerBundle:
    parsed = parse_bundle(document, expected_manifest=expected_manifest)
    entries: list[tuple[str, str]] = [
        (path, parsed.files[path]) for path in sorted(parsed.files)
    ]
    diff = ""
    if baseline_document is not None:
        diff = diff_bundles(
            baseline_document,
            document,
            expected_manifest=parsed.manifest,
        )
    entries.extend(
        [
            ("_skillopt/worker_bundle.diff", diff),
            (
                "_skillopt/worker_bundle_manifest.json",
                json.dumps(parsed.manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
            ),
            ("_skillopt/virtual_skill.md", document),
        ]
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for path, content in entries:
            archive.writestr(_zip_info(path), content.encode("utf-8"))
    return parsed
