# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `site_audit/`. `cli.py` exposes the `site-audit` commands, `pipeline.py` coordinates crawl-to-report execution, and focused analyzers such as `linkgraph.py`, `content_quality.py`, and `compare.py` own individual metrics or views. UI templates live in `ui/`. Generated outputs go under `projects/<domain>/` with `cache/` for intermediate artifacts and `report/` for the final HTML/JSON bundle. Tests belong in `tests/`.

## Build, Test, and Development Commands
Set up a local environment with `python -m venv .venv && source .venv/bin/activate` and `pip install -e .`. Run an audit with `site-audit run example.com`. Start the local viewer with `site-audit serve example.com`. Build a cross-domain report with `site-audit compare visibility.sk gradeta.sk --name preview`. Run checks with `pytest`.

## Coding Style & Naming Conventions
Target Python 3.10+ and follow existing module style: 4-space indentation, type hints on public functions, and small focused helpers over large monoliths. Use `snake_case` for functions, variables, and module names; `PascalCase` for classes such as `PipelineConfig`. Keep CLI flags and output keys descriptive and consistent with existing report terminology.

## Testing Guidelines
Add or update `pytest` tests in `tests/` for behavior changes, especially around CLI flows, payload builders, and analysis helpers. Name files `test_<feature>.py` and keep fixtures narrow so failures are easy to localize. Prefer targeted runs like `pytest tests/test_compare.py` before broader runs.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `Add action plan, paragraph density, content-quality + competitive analyses`. Follow that pattern: start with a verb, describe the visible change, and keep the subject concise. Pull requests should explain the user-facing impact, note any heavy crawl/model implications, and include screenshots or output paths when UI/report rendering changes.

## Generated Data & Safety
Do not hand-edit files inside `projects/<domain>/report/` unless the task is explicitly about generated output. Large crawls and embedding downloads are expensive; prefer cached runs, and use `--clean` only when pipeline logic changes require a full rebuild.
