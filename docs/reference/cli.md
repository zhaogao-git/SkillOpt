# CLI Reference

> **Version note.** This reference tracks `main`. PyPI 0.2.0 does not yet
> include the generic research `openai_compatible` backend, Sleep handoff,
> Sleep support for non-Azure OpenAI-compatible endpoints, or the Sleep
> `--preferences` flag; use a source install from `main` for those features
> until the next release.

## Training

```bash
python scripts/train.py --config <config.yaml> [overrides...]
# Installed equivalent:
skillopt-train --config <config.yaml> [overrides...]
```

### Arguments

| Argument | Description |
|---|---|
| `--config` | Path to YAML config file (required) |
| `--cfg-options key=value [...]` | Override structured config parameters |

### Examples

```bash
# Basic training
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --out_root outputs/searchqa_run

# With overrides
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options optimizer.learning_rate=16 optimizer.lr_scheduler=linear

# With custom initial skill
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options env.skill_init=skills/my_seed.md
```

## Evaluation

```bash
python scripts/eval_only.py --config <config.yaml> --skill <skill.md>
# Installed equivalent:
skillopt-eval --config <config.yaml> --skill <skill.md>
```

### Arguments

| Argument | Description |
|---|---|
| `--config` | Path to YAML config file (required) |
| `--skill` | Path to skill document to evaluate (required) |
| `--split` | `train`, `valid_seen`, `valid_unseen`, or `all` (default) |
| `--cfg-options` | One or more `section.key=value` overrides |

### Examples

```bash
# Evaluate best skill on test set
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa_run/best_skill.md \
  --split valid_unseen

# Evaluate on validation set
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa_run/best_skill.md \
  --split valid_seen
```

`--skill` consumes the artifact produced by training. Unless `--out_root` is
set for evaluation, `eval_only.py` creates a separate timestamped
`outputs/eval_<env>_<model>_<timestamp>/` directory and writes
`eval_summary.json` there; it does not modify the training run directory.

For the generic OpenAI-compatible research backend, select the role backends
explicitly:

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-chat \
    model.target=deepseek-chat
```

For an isolated, tool-free GitHub Copilot CLI optimizer:

```bash
python scripts/train.py \
  --config configs/worker_bundle/template.yaml \
  --cfg-options \
    model.optimizer_backend=copilot_cli \
    model.optimizer=gpt-5.4
```

Replace the template paths locally and configure the external worker runner
before executing it. The backend disables built-in MCPs, custom instructions,
and tools.

## SkillOpt-Sleep

```bash
skillopt-sleep <action> [options]
# Equivalent from a source checkout:
python -m skillopt_sleep <action> [options]
```

Actions are `run`, `dry-run`, `status`, `adopt`, `harvest`, `schedule`, and
`unschedule`. Common options include:

| Argument | Description |
|---|---|
| `--project PATH` | Project to evolve (default: current directory) |
| `--scope invoked\|all` | Harvest this project or all projects |
| `--source claude\|codex\|auto` | Transcript source |
| `--backend mock\|claude\|codex\|copilot\|handoff\|azure_openai` | Replay/optimizer backend |
| `--model NAME` | Backend-specific model override |
| `--preferences TEXT` | House rules supplied to reflection |
| `--lookback-hours N` | Initial transcript lookback; `0` scans all history |
| `--max-sessions N` / `--max-tasks N` | Bound the harvested workload |
| `--target-skill-path PATH` | Explicit skill document to stage/adopt |
| `--tasks-file PATH` | Replay a reviewed task JSON file instead of harvesting |
| `--edit-budget N` | Maximum bounded edits for the night |
| `--progress` / `--json` | Progress or machine-readable output |
| `--auto-adopt` | Apply an accepted staged proposal automatically |

Backend-specific setup for compatible endpoints is documented in
[OpenAI-compatible endpoints for SkillOpt-Sleep](../sleep/openai-compatible-endpoints.md).

## WebUI

```bash
python -m skillopt_webui.app [--port PORT] [--share]
```

| Argument | Default | Description |
|---|---|---|
| `--port` | 7860 | Port number |
| `--host` | `0.0.0.0` | Server bind address |
| `--share` | false | Create public Gradio link |

The default host binds every network interface. Use `--host 127.0.0.1` when
the dashboard should be reachable only from the local machine.
