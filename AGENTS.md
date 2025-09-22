# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `verl/`; version string stored in `verl/version/version`.
- Pytest suites reside in `tests/` with targeted checks under `tests/special_sanity/`.
- Example code and runnable recipes sit in `examples/` and `recipe/`; scripts (e.g., `generate_trainer_config.sh`) are under `scripts/`.
- Documentation is built from `docs/`; container assets are found in `docker/`.

## Build, Test, and Development Commands
- `pip install -e .[test]` installs editable dependencies for local dev and pytest.
- `pytest -q` runs the CPU-friendly suite; select cases via `pytest tests/path::test_name -q`.
- `pre-commit run --all-files` applies `ruff`, `ruff-format`, `mypy`, and trainer config checks.
- `make -C docs html` builds the Sphinx docs after `pip install -r docs/requirements-docs.txt`.

## Coding Style & Naming Conventions
- Python code uses 4-space indents, line length 120, and explicit type hints when practical.
- Follow `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_CASE` for constants.
- Keep imports sorted; rely on `ruff` hooks for linting and formatting enforcement.

## Testing Guidelines
- Tests must live under `tests/` and be named `test_*.py`; mark resource-heavy cases to skip gracefully (see existing `@pytest.mark.skipif`).
- Use `pytest-asyncio` fixtures as needed and keep parametrized coverage for model variants.
- Before submitting, run `pytest -q`; document any skipped GPU-dependent checks in the PR notes.

## Commit & Pull Request Guidelines
- Commit messages follow `[scope] type: summary` (e.g., `[trainer] fix: handle None states`). Link related issues in the body (`Fixes #123`).
- PRs should explain rationale, note testing (CPU/GPU/local), mention config or doc updates, and ensure `pre-commit` passes.
- Regenerate trainer configs with `scripts/generate_trainer_config.sh` when touching `verl/trainer/config/`.

## Security & Configuration Tips
- Never commit secrets or datasets; rely on environment variables and `.gitignore`d paths.
- Validate new trainer settings locally before sharing, and document required env vars in PR descriptions.
