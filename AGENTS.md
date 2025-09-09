# Repository Guidelines

## Project Structure & Module Organization
- Source code: `verl/` (core library), version at `verl/version/version`.
- Tests: `tests/` (pytest), special sanity checks in `tests/special_sanity/`.
- Examples & recipes: `examples/`, `recipe/`.
- Tooling & scripts: `scripts/` (e.g., `generate_trainer_config.sh`).
- Docs & site: `docs/` (Sphinx), Docker assets in `docker/`.

## Build, Test, and Development Commands
- Install (editable) with extras:
  - `pip install -e .[test]` (core dev + tests)
  - `pip install -e .[test,vllm]` or `.[test,sglang]` for engine-specific work
- Run tests: `pytest -q` (GPU-heavy tests auto-skip if unsupported). Example: `pytest tests/utils/test_torch_functional.py::test_allreduce -q`.
- Lint/format/type-check via pre-commit:
  - `pre-commit install`
  - `pre-commit run --all-files`
  - Common hooks: `ruff`, `ruff-format`, `mypy`, `autogen-trainer-cfg`, docstrings/license checks.
- Build docs:
  - `pip install -r docs/requirements-docs.txt`
  - `make -C docs html`

## Coding Style & Naming Conventions
- Python, 4-space indent, line length 120 (ruff).
- Use type hints where practical; mypy configured with selective overrides.
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_CASE` for constants.
- Keep imports sorted (ruff/isort). Prefer explicit exports over wildcard except where justified by config.

## Testing Guidelines
- Framework: `pytest` (+ `pytest-asyncio` via `[test]`).
- Place tests under `tests/`; name files `test_*.py`; use `parametrize` and `asyncio` markers as needed.
- Resource-dependent tests should skip cleanly (see existing `@pytest.mark.skipif` patterns and envs like `SANDBOX_FUSION_URL`).

## Commit & Pull Request Guidelines
- Commit style: `[scope] type: summary`, imperative mood. Examples: `[trainer] fix: handle None states`, `[sglang] feat: add native server`.
- Link issues in body (e.g., `Fixes #123`). Keep commits focused.
- PRs must include: clear description, rationale, testing notes (CPU/GPU/local), config changes, and doc updates if user-facing. Ensure pre-commit passes and CI is green.

## Tips & Notes
- Trainer configs live under `verl/trainer/config/`; regenerate generated YAML via `scripts/generate_trainer_config.sh` (runs in pre-commit).
- Do not commit secrets or large datasets. Use environment variables and `.gitignore`d paths for local data.
