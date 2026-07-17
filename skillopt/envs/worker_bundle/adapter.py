"""SkillOpt adapter for generic virtual worker bundles."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.worker_bundle.dataloader import WorkerBundleDataLoader
from skillopt.envs.worker_bundle.rollout import run_batch
from skillopt.worker_bundle import (
    WorkerBundleManifest,
    bundle_edit_stats,
    export_bundle_zip,
    load_manifest,
    materialize_bundle,
    serialize_bundle,
    validate_bundle_edit_limits,
)


class WorkerBundleAdapter(EnvAdapter):
    """Evaluate virtual worker bundles through a configured external executable."""

    def __init__(
        self,
        split_dir: str = "",
        split_mode: str = "split_dir",
        allow_test_split: bool = False,
        bundle_manifest: str = "",
        bundle_base_dir: str = "",
        runner_path: str = "",
        runner_env_var: str = "WORKER_BUNDLE_RUNNER",
        runner_args: list[str] | None = None,
        runner_timeout: int = 120,
        max_files_changed: int = 3,
        max_edit_operations: int = 8,
        max_added_tokens: int = 2000,
        max_removed_tokens: int = 2000,
        analyst_workers: int = 4,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 2,
        seed: int = 42,
        limit: int = 0,
    ) -> None:
        self.bundle_manifest_path = bundle_manifest
        self.bundle_base_dir = bundle_base_dir
        self.runner_path = runner_path
        self.runner_env_var = runner_env_var
        self.runner_args = list(runner_args or [])
        self.runner_timeout = int(runner_timeout)
        self.max_files_changed = int(max_files_changed)
        self.max_edit_operations = int(max_edit_operations)
        self.max_added_tokens = int(max_added_tokens)
        self.max_removed_tokens = int(max_removed_tokens)
        self.analyst_workers = int(analyst_workers)
        self.failure_only = bool(failure_only)
        self.minibatch_size = int(minibatch_size)
        self.edit_budget = int(edit_budget)
        self.manifest: WorkerBundleManifest | None = None
        self.dataloader = WorkerBundleDataLoader(
            split_dir=split_dir,
            split_mode=split_mode,
            allow_test_split=allow_test_split,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        if str(cfg.get("skill_update_mode", "patch")).strip().lower() != "patch":
            raise ValueError("worker_bundle POC requires skill_update_mode='patch'")
        for key in (
            "use_slow_update",
            "use_meta_skill",
            "use_skill_aware_reflection",
            "use_semantic_density",
        ):
            if cfg.get(key, False):
                raise ValueError(f"worker_bundle POC requires {key}=false")
        self.dataloader.setup(cfg)
        if not self.bundle_manifest_path:
            self.bundle_manifest_path = str(cfg.get("bundle_manifest") or "")
        if not self.bundle_base_dir:
            self.bundle_base_dir = str(cfg.get("bundle_base_dir") or "")
        if not self.bundle_manifest_path:
            raise ValueError("worker_bundle requires bundle_manifest")
        if not self.bundle_base_dir:
            raise ValueError("worker_bundle requires bundle_base_dir")
        self.manifest = load_manifest(self.bundle_manifest_path)
        if not Path(self.bundle_base_dir).is_dir():
            raise ValueError(f"worker_bundle bundle_base_dir does not exist: {self.bundle_base_dir}")

        configured_runner = self.runner_path or os.environ.get(self.runner_env_var, "")
        if not configured_runner:
            raise ValueError(
                "worker_bundle requires runner_path or the "
                f"{self.runner_env_var!r} environment variable"
            )
        resolved_runner = shutil.which(configured_runner)
        if resolved_runner is None and Path(configured_runner).is_file():
            resolved_runner = str(Path(configured_runner).resolve())
        if resolved_runner is None:
            raise ValueError(f"worker_bundle runner executable was not found: {configured_runner}")
        self.runner_path = resolved_runner
        if self.runner_timeout <= 0:
            raise ValueError("worker_bundle runner_timeout must be positive")
        if not all(isinstance(arg, str) for arg in self.runner_args):
            raise ValueError("worker_bundle runner_args must be a list of strings")
        for label in (
            "max_files_changed",
            "max_edit_operations",
            "max_added_tokens",
            "max_removed_tokens",
        ):
            if getattr(self, label) < 0:
                raise ValueError(f"worker_bundle {label} must be non-negative")

    def get_dataloader(self) -> WorkerBundleDataLoader:
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        if self.manifest is None:
            raise RuntimeError("worker_bundle adapter.setup() must be called before rollout")
        return run_batch(
            items=list(env_manager),
            skill_content=skill_content,
            out_root=out_dir,
            manifest=self.manifest,
            base_dir=self.bundle_base_dir,
            runner_path=self.runner_path,
            runner_args=self.runner_args,
            runner_timeout=self.runner_timeout,
            max_files_changed=self.max_files_changed,
            max_edit_operations=self.max_edit_operations,
            max_added_tokens=self.max_added_tokens,
            max_removed_tokens=self.max_removed_tokens,
        )

    def get_task_types(self) -> list[str]:
        task_types: list[str] = []
        for item in self.dataloader.train_items + self.dataloader.val_items + self.dataloader.test_items:
            task_type = str(item.get("task_type") or "worker_bundle")
            if task_type not in task_types:
                task_types.append(task_type)
        return task_types or ["worker_bundle"]

    def export_best_bundle(self, skill_content: str, out_root: str) -> None:
        if self.manifest is None:
            raise RuntimeError("worker_bundle adapter.setup() must be called before export")
        base_dir = Path(self.bundle_base_dir)
        baseline_files = {
            entry.path: base_dir.joinpath(*entry.path.split("/")).read_text(encoding="utf-8")
            for entry in self.manifest.files
        }
        baseline_document = serialize_bundle(baseline_files, self.manifest)
        stats = bundle_edit_stats(
            baseline_document,
            skill_content,
            expected_manifest=self.manifest,
        )
        validate_bundle_edit_limits(
            stats,
            max_files_changed=self.max_files_changed,
            max_edit_operations=self.max_edit_operations,
            max_added_tokens=self.max_added_tokens,
            max_removed_tokens=self.max_removed_tokens,
        )
        output = Path(out_root)
        parsed = materialize_bundle(
            skill_content,
            output / "best_worker_bundle",
            expected_manifest=self.manifest,
            precondition_dir=base_dir,
        )
        export_bundle_zip(
            skill_content,
            output / "best_worker_bundle.zip",
            expected_manifest=self.manifest,
            baseline_document=baseline_document,
        )
        (output / "best_worker_bundle_metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate_hash": parsed.candidate_hash,
                    "manifest_hash": parsed.manifest_hash,
                    "edit_stats": stats.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
