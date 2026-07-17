# Worker Bundle POC

`worker_bundle` is a generic proof of concept for optimizing a fixed set of
worker instruction files without refactoring SkillOpt's core trainable state.
The trainer still sees one Markdown string. That string is a canonical virtual
document containing an immutable manifest and one section per allowed file.

This POC is intentionally limited to mutable instruction/strategy files.
Knowledge sources, tools, credentials, runtime configuration, and external
systems remain outside the virtual document.

## Safety and integrity contract

The manifest fixes each allowed relative path, role, and baseline SHA-256.
Serialization sorts paths and normalizes line endings. Parsing rejects:

- changed or missing headers and boundary markers;
- changed manifest hashes or manifests;
- duplicate, missing, or unknown file sections;
- absolute paths, traversal, backslashes, and symlinks;
- changed path/role/base-hash markers;
- baseline precondition mismatches.

Candidates cannot add or delete files. `materialize_bundle()` validates first,
writes a sibling staging directory, and swaps it into the output location with
rollback if the swap fails. Helpers also expose deterministic candidate and
manifest hashes, unified file diffs, and reproducible ZIP exports.

## External runner contract

The benchmark loads reviewed, pre-split YAML task contracts:

```text
task-root/
├── train/*.yaml
├── val/*.yaml
└── test/*.yaml
```

Each task needs `id` or `taskId` plus `prompt` (or
`question`/`task_description`). IDs must be unique across all loaded splits.
Other fields are passed unchanged to the runner. The adapter resolves the runner from
`env.runner_path` or the environment variable named by
`env.runner_env_var` (default `WORKER_BUNDLE_RUNNER`).

For each task it invokes this argument array, never a shell:

```text
<runner> <runner_args...> --request <runner_request.json>
```

The request JSON contains:

```json
{
  "schema_version": 1,
  "task": {"id": "example", "prompt": "Handle this request."},
  "bundle_dir": "/materialized/candidate",
  "candidate_hash": "<sha256>",
  "manifest_hash": "<sha256>"
}
```

The runner writes one JSON object to stdout:

```json
{
  "answer": "Final answer text",
  "hard": 1,
  "soft": 0.8,
  "metadata": {"grader_version": "example"}
}
```

The adapter persists the request, reflection `conversation.json`, rollout
fields, candidate directory, virtual document, manifest, diff, metadata, and
candidate ZIP. It has no domain-specific knowledge.

Generated ZIP metadata is stored below the reserved `_skillopt/` namespace so
it cannot replace worker files.

Before materialization or runner invocation, the adapter computes changed-file,
edit-operation, added/removed-line, and approximate added/removed-token counts.
`max_files_changed`, `max_edit_operations`, `max_added_tokens`, and
`max_removed_tokens` are hard limits. The statistics and limits are recorded in
`candidate_metadata.json`.

At trainer completion, the adapter preserves upstream `best_skill.md` and also
writes `best_worker_bundle/`, `best_worker_bundle.zip`, and
`best_worker_bundle_metadata.json`.

## Configuration

Copy `configs/worker_bundle/template.yaml` and replace its neutral placeholder
paths locally. Generate a manifest from the baseline mutable files with
`build_manifest()` and serialize the initial virtual document with
`serialize_bundle()`.

Keep `optimizer.skill_update_mode: patch`,
`optimizer.use_slow_update: false`, `optimizer.use_meta_skill: false`, and
`evaluation.use_semantic_density: false` for this POC. Leave
`evaluation.eval_test: false` until a separate process is authorized to open a
sealed test split.

The adapter loads only `train/` and `val/` during normal setup. It does not read
`test/` unless `env.allow_test_split: true` is set in a separately authorized
evaluation process. Missing or encrypted test data therefore cannot be opened
accidentally during training.

## Intentional limitations

- The core state remains one string; this is not a `TrainableState` refactor.
- Invalid optimizer edits fail validation during rollout rather than being
  represented as typed multi-file patch operations.
- Files cannot be added, deleted, renamed, or converted into tools.
- External runner/runtime fidelity, data governance, and grader quality remain
  the caller's responsibility.
