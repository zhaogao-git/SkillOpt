"""Tool-free GitHub Copilot CLI backend for SkillOpt model calls."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from skillopt.model.backend_config import get_copilot_cli_config
from skillopt.model.common import CompatAssistantMessage, TokenTracker

OPTIMIZER_DEPLOYMENT = os.environ.get("OPTIMIZER_DEPLOYMENT", "gpt-5.4")
TARGET_DEPLOYMENT = os.environ.get("TARGET_DEPLOYMENT", "gpt-5.4")
tracker = TokenTracker()


def _validate_tool_free_messages(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> None:
    if tools or tool_choice not in (None, "none"):
        raise ValueError("copilot_cli is tool-free and does not accept tools or tool_choice")
    for message in messages:
        if message.get("role") == "tool" or message.get("tool_calls"):
            raise ValueError("copilot_cli is tool-free and does not accept tool messages or tool calls")


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    system_parts: list[str] = []
    conversation: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("copilot_cli only accepts text message content")
        if role == "system":
            if content.strip():
                system_parts.append(content.strip())
            continue
        conversation.append(f"{role.capitalize()}: {content.strip()}")
    if len(conversation) == 1 and conversation[0].startswith("User: "):
        user_text = conversation[0][len("User: ") :]
        return "\n\n".join([*system_parts, user_text])
    parts = list(system_parts)
    if conversation:
        parts.append("Conversation:\n" + "\n".join(conversation))
    return "\n\n".join(parts)


def _parse_jsonl(stdout: str) -> str:
    parts: list[str] = []
    saw_json = False
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Copilot CLI returned non-JSON output") from exc
        if not isinstance(event, dict):
            raise RuntimeError("Copilot CLI returned a non-object JSON event")
        saw_json = True
        event_type = str(event.get("type") or "")
        data = event.get("data") or {}
        if event_type.startswith("tool.") or (
            isinstance(data, dict)
            and (
                data.get("tool_calls")
                or data.get("toolCall")
                or data.get("toolRequests")
            )
        ):
            raise RuntimeError("Copilot CLI requested a tool in tool-free mode")
        if event_type == "assistant.message":
            content = data.get("content") if isinstance(data, dict) else None
            if not isinstance(content, str):
                raise RuntimeError("Copilot CLI assistant.message content must be text")
            if content:
                parts.append(content)
    if not saw_json:
        raise RuntimeError("Copilot CLI returned no JSON events")
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError("Copilot CLI returned an empty assistant response")
    return text


def _run_copilot(
    *,
    model: str,
    prompt: str,
    timeout: int | None,
) -> str:
    config = get_copilot_cli_config()
    copilot_home = Path(str(config["copilot_home"]))
    cwd = Path(str(config["cwd"]))
    if copilot_home.is_symlink():
        raise RuntimeError(f"copilot_cli COPILOT_HOME must not be a symlink: {copilot_home}")
    copilot_home.mkdir(parents=True, exist_ok=True)
    if cwd.is_symlink() or not cwd.is_dir():
        raise RuntimeError(f"copilot_cli cwd must be an existing non-symlink directory: {cwd}")

    command = [
        str(config["path"]),
        "--prompt",
        prompt,
        "--output-format",
        "json",
        "--stream",
        "off",
        "--no-color",
        "--log-level",
        "none",
        "--disable-builtin-mcps",
        "--no-custom-instructions",
        "--no-ask-user",
        "--available-tools=",
        "--model",
        model,
        "--reasoning-effort",
        str(config["reasoning_effort"]),
        "--context",
        str(config["context"]),
        "-C",
        str(cwd),
    ]
    env = os.environ.copy()
    env.pop("COPILOT_ALLOW_ALL", None)
    env["COPILOT_HOME"] = str(copilot_home)
    effective_timeout = timeout if timeout is not None else int(config["timeout_seconds"])
    proc = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=effective_timeout,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no error output"
        raise RuntimeError(f"Copilot CLI exited with code {proc.returncode}: {detail[:1000]}")
    return _parse_jsonl(proc.stdout)


def _chat_messages_impl(
    model: str,
    messages: list[dict[str, Any]],
    retries: int,
    stage: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict[str, int]]:
    _validate_tool_free_messages(messages, tools=tools, tool_choice=tool_choice)
    if retries <= 0:
        raise ValueError("copilot_cli retries must be positive")
    prompt = _messages_to_prompt(messages)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            text = _run_copilot(model=model, prompt=prompt, timeout=timeout)
            tracker.record(stage, 0, 0)
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if return_message:
                return CompatAssistantMessage(content=text), usage
            return text, usage
        except subprocess.TimeoutExpired:
            last_error = RuntimeError("Copilot CLI timed out")
        except (OSError, RuntimeError) as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Copilot CLI call failed after {retries} retries: {last_error}")


def chat_with_model(
    model: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    del max_completion_tokens
    return _chat_messages_impl(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        retries,
        stage,
        timeout=timeout,
    )


def chat_messages_with_model(
    model: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict[str, int]]:
    del max_completion_tokens
    return _chat_messages_impl(
        model,
        messages,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    return chat_with_model(
        OPTIMIZER_DEPLOYMENT,
        system,
        user,
        max_completion_tokens,
        retries,
        stage,
        timeout,
    )


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    return chat_with_model(
        TARGET_DEPLOYMENT,
        system,
        user,
        max_completion_tokens,
        retries,
        stage,
        timeout,
    )


def chat_with_deployment(
    deployment: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    timeout: int | None = None,
) -> tuple[str, dict[str, int]]:
    return chat_with_model(
        deployment,
        system,
        user,
        max_completion_tokens,
        retries,
        stage,
        timeout,
    )


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict[str, int]]:
    return chat_messages_with_model(
        OPTIMIZER_DEPLOYMENT,
        messages,
        max_completion_tokens,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def chat_target_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict[str, int]]:
    return chat_messages_with_model(
        TARGET_DEPLOYMENT,
        messages,
        max_completion_tokens,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def chat_messages_with_deployment(
    deployment: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict[str, int]]:
    return chat_messages_with_model(
        deployment,
        messages,
        max_completion_tokens,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_target_deployment(deployment: str) -> None:
    global TARGET_DEPLOYMENT
    TARGET_DEPLOYMENT = deployment


def set_optimizer_deployment(deployment: str) -> None:
    global OPTIMIZER_DEPLOYMENT
    OPTIMIZER_DEPLOYMENT = deployment


def set_reasoning_effort(effort: str | None) -> None:
    value = str(effort).strip().lower() if effort else ""
    if value in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        from skillopt.model.backend_config import configure_copilot_cli

        configure_copilot_cli(reasoning_effort=value)
