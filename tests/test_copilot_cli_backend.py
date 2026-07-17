from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


class _OpenAIClientStub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def _install_openai_stub() -> None:
    if "openai" in sys.modules or importlib.util.find_spec("openai") is not None:
        return
    openai_stub = types.ModuleType("openai")
    openai_stub.AzureOpenAI = _OpenAIClientStub
    openai_stub.OpenAI = _OpenAIClientStub
    sys.modules["openai"] = openai_stub


def _import_model_modules() -> tuple[Any, Any, Any]:
    _install_openai_stub()
    import skillopt.model as model_module
    from skillopt.model import backend_config, copilot_cli_backend

    return model_module, backend_config, copilot_cli_backend


@pytest.fixture(autouse=True)
def isolate_backend_state() -> Iterator[tuple[Any, Any, Any]]:
    model_module, backend_config, copilot_backend = _import_model_modules()
    optimizer_backend = backend_config.get_optimizer_backend()
    target_backend = backend_config.get_target_backend()
    config = backend_config.get_copilot_cli_config()
    optimizer_model = copilot_backend.OPTIMIZER_DEPLOYMENT
    target_model = copilot_backend.TARGET_DEPLOYMENT
    copilot_backend.reset_token_tracker()
    yield model_module, backend_config, copilot_backend
    copilot_backend.reset_token_tracker()
    backend_config.configure_copilot_cli(**config)
    backend_config.set_optimizer_backend(optimizer_backend)
    backend_config.set_target_backend(target_backend)
    copilot_backend.set_optimizer_deployment(optimizer_model)
    copilot_backend.set_target_deployment(target_model)


def _write_fake_copilot(root: Path) -> Path:
    executable = root / "fake-copilot.py"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

log_path = Path(os.environ["FAKE_COPILOT_LOG"])
count_path = Path(os.environ["FAKE_COPILOT_COUNT"])
count = int(count_path.read_text(encoding="utf-8")) if count_path.exists() else 0
count += 1
count_path.write_text(str(count), encoding="utf-8")
log_path.write_text(json.dumps({
    "argv": sys.argv[1:],
    "copilot_home": os.environ.get("COPILOT_HOME"),
    "cwd": os.getcwd(),
}, sort_keys=True), encoding="utf-8")
mode = os.environ.get("FAKE_COPILOT_MODE", "success")
if mode == "fail_once" and count == 1:
    print("first failure", file=sys.stderr)
    raise SystemExit(7)
if mode == "tool":
    print(json.dumps({"type": "tool.execution_start", "data": {"name": "shell"}}))
    print(json.dumps({"type": "assistant.message", "data": {"content": "should be rejected"}}))
    raise SystemExit(0)
if mode == "invalid":
    print("not-json")
    raise SystemExit(0)
print(json.dumps({"type": "assistant.message", "data": {"content": "first"}}))
print(json.dumps({"type": "assistant.message", "data": {"content": "second"}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _configure_fake(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_config: Any,
    *,
    mode: str = "success",
) -> tuple[Path, Path, Path]:
    executable = _write_fake_copilot(root)
    log_path = root / "copilot-log.json"
    count_path = root / "copilot-count.txt"
    home = root / "copilot-home"
    cwd = root / "copilot-cwd"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("FAKE_COPILOT_LOG", str(log_path))
    monkeypatch.setenv("FAKE_COPILOT_COUNT", str(count_path))
    monkeypatch.setenv("FAKE_COPILOT_MODE", mode)
    backend_config.configure_copilot_cli(
        path=str(executable),
        reasoning_effort="high",
        context="long_context",
        copilot_home=str(home),
        cwd=str(cwd),
        timeout_seconds=9,
    )
    return log_path, count_path, home


def test_copilot_cli_command_is_fixed_tool_free_and_parses_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolate_backend_state: tuple[Any, Any, Any],
) -> None:
    model_module, backend_config, copilot_backend = isolate_backend_state
    log_path, _count_path, home = _configure_fake(tmp_path, monkeypatch, backend_config)
    backend_config.set_optimizer_backend("copilot_cli")
    copilot_backend.set_optimizer_deployment("test-model")

    text, usage = model_module.chat_optimizer(
        "System instructions",
        "User request",
        retries=1,
        timeout=5,
        stage="optimizer-test",
    )

    assert text == "first\nsecond"
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    log = json.loads(log_path.read_text(encoding="utf-8"))
    argv = log["argv"]
    assert argv[argv.index("--prompt") + 1] == "System instructions\n\nUser request"
    assert argv[argv.index("--model") + 1] == "test-model"
    assert argv[argv.index("--reasoning-effort") + 1] == "high"
    assert argv[argv.index("--context") + 1] == "long_context"
    assert "--disable-builtin-mcps" in argv
    assert "--no-custom-instructions" in argv
    assert "--available-tools=" in argv
    assert "--allow-all-tools" not in argv
    assert log["copilot_home"] == str(home)
    assert Path(log["cwd"]) == tmp_path / "copilot-cwd"

    summary = model_module.get_token_summary()
    assert summary["optimizer-test"] == {
        "calls": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    model_module.reset_token_tracker()
    assert "optimizer-test" not in model_module.get_token_summary()


def test_copilot_cli_retries_failed_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolate_backend_state: tuple[Any, Any, Any],
) -> None:
    model_module, backend_config, copilot_backend = isolate_backend_state
    _log_path, count_path, _home = _configure_fake(tmp_path, monkeypatch, backend_config, mode="fail_once")
    monkeypatch.setattr(copilot_backend.time, "sleep", lambda _seconds: None)
    backend_config.set_target_backend("copilot_cli")
    copilot_backend.set_target_deployment("test-model")

    text, _usage = model_module.chat_target("system", "user", retries=2)

    assert text == "first\nsecond"
    assert count_path.read_text(encoding="utf-8") == "2"


def test_copilot_cli_rejects_tool_inputs_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolate_backend_state: tuple[Any, Any, Any],
) -> None:
    model_module, backend_config, _copilot_backend = isolate_backend_state
    log_path, _count_path, _home = _configure_fake(tmp_path, monkeypatch, backend_config)
    backend_config.set_optimizer_backend("copilot_cli")

    with pytest.raises(ValueError, match="tool-free"):
        model_module.chat_optimizer_messages(
            [{"role": "user", "content": "request"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            retries=1,
        )
    with pytest.raises(ValueError, match="tool-free"):
        model_module.chat_optimizer_messages(
            [{"role": "tool", "content": "result", "tool_call_id": "call-1"}],
            retries=1,
        )
    assert not log_path.exists()


def test_copilot_cli_rejects_output_tool_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolate_backend_state: tuple[Any, Any, Any],
) -> None:
    model_module, backend_config, _copilot_backend = isolate_backend_state
    _configure_fake(tmp_path, monkeypatch, backend_config, mode="tool")
    backend_config.set_target_backend("copilot_cli")

    with pytest.raises(RuntimeError, match="requested a tool"):
        model_module.chat_target("system", "user", retries=1)


def test_copilot_cli_message_variants_and_return_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolate_backend_state: tuple[Any, Any, Any],
) -> None:
    model_module, backend_config, _copilot_backend = isolate_backend_state
    _configure_fake(tmp_path, monkeypatch, backend_config)
    backend_config.set_target_backend("copilot_cli")

    message, usage = model_module.chat_target_messages(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first user"},
            {"role": "assistant", "content": "previous"},
            {"role": "user", "content": "latest"},
        ],
        return_message=True,
        retries=1,
    )

    assert message.content == "first\nsecond"
    assert message.tool_calls == []
    assert usage["total_tokens"] == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"reasoning_effort": "extreme"}, "reasoning"),
        ({"context": "huge"}, "context"),
        ({"timeout_seconds": 0}, "timeout"),
        ({"copilot_home": ""}, "COPILOT_HOME"),
        ({"cwd": ""}, "cwd"),
    ],
)
def test_copilot_cli_configuration_validation(
    isolate_backend_state: tuple[Any, Any, Any],
    kwargs: dict[str, Any],
    match: str,
) -> None:
    _model_module, backend_config, _copilot_backend = isolate_backend_state
    config = {
        "path": "copilot",
        "reasoning_effort": "medium",
        "context": "default",
        "copilot_home": "/generic/copilot-home",
        "cwd": "/generic/workspace",
        "timeout_seconds": 120,
    }
    config.update(kwargs)

    with pytest.raises(ValueError, match=match):
        backend_config.configure_copilot_cli(**config)


def test_global_reasoning_effort_ignores_values_unsupported_by_copilot(
    isolate_backend_state: tuple[Any, Any, Any],
) -> None:
    model_module, backend_config, _copilot_backend = isolate_backend_state
    original = backend_config.get_copilot_cli_config()["reasoning_effort"]

    model_module.set_reasoning_effort("provider-specific")

    assert backend_config.get_copilot_cli_config()["reasoning_effort"] == original


def test_copilot_cli_alias_default_model_and_config_flattening(
    isolate_backend_state: tuple[Any, Any, Any],
) -> None:
    model_module, backend_config, _copilot_backend = isolate_backend_state
    from skillopt.config import flatten_config
    from skillopt.model.common import default_model_for_backend, normalize_backend_name

    assert normalize_backend_name("copilot") == "copilot_cli"
    assert normalize_backend_name("copilot-cli") == "copilot_cli"
    assert default_model_for_backend("copilot_cli") == "gpt-5.4"
    assert model_module.set_backend("copilot") == "copilot_cli"
    assert backend_config.get_optimizer_backend() == "copilot_cli"
    assert backend_config.get_target_backend() == "copilot_cli"
    assert model_module.get_backend_name() == "copilot_cli"

    flat = flatten_config(
        {
            "model": {
                "copilot_cli_path": "copilot",
                "copilot_cli_reasoning_effort": "high",
                "copilot_cli_context": "long_context",
                "copilot_cli_home": ".skillopt/copilot-home",
                "copilot_cli_cwd": ".",
                "copilot_cli_timeout_seconds": 90,
            },
            "env": {"name": "worker_bundle"},
        }
    )
    assert flat == {
        "copilot_cli_path": "copilot",
        "copilot_cli_reasoning_effort": "high",
        "copilot_cli_context": "long_context",
        "copilot_cli_home": ".skillopt/copilot-home",
        "copilot_cli_cwd": ".",
        "copilot_cli_timeout_seconds": 90,
        "env": "worker_bundle",
    }
