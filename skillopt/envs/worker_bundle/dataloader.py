"""Pre-split YAML task loader for the worker_bundle benchmark."""
from __future__ import annotations

import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from skillopt.datasets.base import SplitDataLoader

_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _parse_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{label} must be a boolean")


def _normalize_yaml_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _normalize_yaml_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_yaml_value(item) for item in value]
    return value


class WorkerBundleDataLoader(SplitDataLoader):
    """Load generic worker task contracts from train/val/test YAML files."""

    def __init__(
        self,
        *,
        split_dir: str = "",
        split_mode: str = "split_dir",
        allow_test_split: bool = False,
        seed: int = 42,
        limit: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            split_dir=split_dir,
            split_mode=split_mode,
            seed=seed,
            limit=limit,
            **kwargs,
        )
        self.allow_test_split = _parse_bool(
            allow_test_split,
            label="allow_test_split",
        )
        self._loaded_ids: dict[str, str] = {}

    def setup(self, cfg: dict) -> None:
        mode = str(self.split_mode or cfg.get("split_mode") or "split_dir").strip().lower()
        if mode != "split_dir":
            raise ValueError("WorkerBundleDataLoader requires split_mode='split_dir'")
        self.split_mode = mode
        if not self.split_dir:
            self.split_dir = str(cfg.get("split_dir") or "")
        if not self.split_dir:
            raise ValueError("WorkerBundleDataLoader requires split_dir")
        if "allow_test_split" in cfg:
            self.allow_test_split = _parse_bool(
                cfg["allow_test_split"],
                label="allow_test_split",
            )
        self._splits = {}
        self._loaded_ids = {}
        self._load_split("train")
        self._load_split("val")
        print(
            f"  [{type(self).__name__}] "
            f"train={len(self.train_items)} val={len(self.val_items)} test=sealed "
            f"(from {self.split_dir})"
        )

    def _load_split(self, name: str) -> None:
        split_path = os.path.join(self.split_dir, name)
        if not os.path.isdir(split_path):
            raise ValueError(f"Missing '{name}/' subdirectory in split_dir: {self.split_dir}")
        items = self.load_split_items(split_path)
        if self.limit:
            items = items[: self.limit]
        for item in items:
            task_id = str(item["id"])
            previous_split = self._loaded_ids.get(task_id)
            if previous_split is not None:
                raise ValueError(
                    f"Duplicate worker_bundle task id across worker_bundle splits: "
                    f"{task_id!r} appears in {previous_split!r} and {name!r}"
                )
            self._loaded_ids[task_id] = name
        self._splits[name] = items

    def get_split_items(self, split: str) -> list[dict]:
        canonical = {
            "train": "train",
            "valid_seen": "val",
            "selection": "val",
            "val": "val",
            "valid_unseen": "test",
            "test": "test",
        }.get(split, split)
        if canonical not in {"train", "val", "test"}:
            raise ValueError(f"Unknown worker_bundle split: {split!r}")
        if canonical == "test" and "test" not in self._splits:
            if not self.allow_test_split:
                raise PermissionError(
                    "worker_bundle sealed test split access is disabled; "
                    "set allow_test_split=true only in an authorized evaluation process"
                )
            self._load_split("test")
        return list(self._splits.get(canonical, self.val_items))

    def load_split_items(self, split_path: str) -> list[dict]:
        yaml_files = sorted(Path(split_path).glob("*.yaml"))
        yaml_files += sorted(Path(split_path).glob("*.yml"))
        if not yaml_files:
            raise FileNotFoundError(f"No YAML task contract file found in {split_path}")

        items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for path in yaml_files:
            with path.open(encoding="utf-8") as handle:
                value = yaml.safe_load(handle)
            if isinstance(value, dict) and "tasks" in value:
                raw_items = value["tasks"]
            elif isinstance(value, list):
                raw_items = value
            elif isinstance(value, dict):
                raw_items = [value]
            else:
                raise ValueError(f"Worker bundle task file must contain a mapping or list: {path}")
            if not isinstance(raw_items, list):
                raise ValueError(f"Worker bundle tasks must be a list: {path}")
            for raw in raw_items:
                if not isinstance(raw, dict):
                    raise ValueError(f"Worker bundle task entries must be mappings: {path}")
                item = _normalize_yaml_value(raw)
                raw_task_id = item.get("id")
                if raw_task_id is None:
                    raw_task_id = item.get("taskId")
                if raw_task_id is None or isinstance(raw_task_id, (dict, list, bool)):
                    raise ValueError(f"Worker bundle task id is required: {path}")
                task_id = str(raw_task_id).strip()
                if not task_id:
                    raise ValueError(f"Worker bundle task id is required: {path}")
                if not _SAFE_TASK_ID.fullmatch(task_id):
                    raise ValueError(
                        f"Worker bundle task id must be filesystem-safe, got {task_id!r}"
                    )
                if task_id in seen_ids:
                    raise ValueError(f"Duplicate worker_bundle task id: {task_id}")
                prompt = str(
                    item.get("prompt")
                    or item.get("question")
                    or item.get("task_description")
                    or ""
                ).strip()
                if not prompt:
                    raise ValueError(f"Worker bundle task prompt is required for id {task_id!r}")
                seen_ids.add(task_id)
                item["id"] = task_id
                item["prompt"] = prompt
                item.setdefault("task_type", "worker_bundle")
                items.append(item)
        return items
