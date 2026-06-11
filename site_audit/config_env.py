"""Local .env-backed defaults for CLI settings."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any


def load_dotenv(path: Path | str | None = None) -> Path | None:
    env_path = Path(path) if path else _find_dotenv()
    if env_path is None or not env_path.is_file():
        return env_path
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path


def _find_dotenv(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return cur / ".env"


def env_names(command: str | None, dest: str) -> list[str]:
    key = dest.upper()
    names = []
    if command:
        command_key = re.sub(r"[^A-Z0-9]+", "_", command.upper()).strip("_")
        names.append(f"SITE_AUDIT_{command_key}_{key}")
    names.append(f"SITE_AUDIT_{key}")
    return names


def collect_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    actions: list[argparse.Action] = []
    seen: set[int] = set()

    def visit(p: argparse.ArgumentParser) -> None:
        for action in p._actions:
            if id(action) not in seen:
                seen.add(id(action))
                actions.append(action)
            if isinstance(action, argparse._SubParsersAction):
                for subparser in action.choices.values():
                    visit(subparser)

    visit(parser)
    return actions


def cli_supplied_options(argv: list[str]) -> set[str]:
    supplied: set[str] = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        supplied.add(option)
    return supplied


def apply_env_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str]) -> None:
    load_dotenv()
    supplied = cli_supplied_options(argv)
    command = getattr(args, "command", None)
    for action in collect_actions(parser):
        if isinstance(action, argparse._SubParsersAction):
            continue
        if not action.dest or action.dest in {"help", argparse.SUPPRESS}:
            continue
        if any(opt in supplied for opt in action.option_strings):
            continue
        if not hasattr(args, action.dest):
            continue
        if not action.option_strings and getattr(args, action.dest) not in (None, "", []):
            continue
        raw = _first_env(env_names(command, action.dest))
        if raw is None:
            continue
        if raw == "":
            continue
        setattr(args, action.dest, _coerce_value(raw, action))


def _first_env(names: list[str]) -> str | None:
    for name in names:
        if name in os.environ:
            return os.environ[name]
    return None


def _coerce_value(raw: str, action: argparse.Action) -> Any:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
        return _to_bool(raw)
    if isinstance(action, argparse._AppendAction):
        return _to_list(raw)
    if isinstance(action, argparse._CountAction):
        return int(raw or "0")
    if action.nargs in {"+", "*"}:
        return _to_list(raw)
    if action.type:
        return action.type(raw)
    return raw


def _to_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_list(raw: str) -> list[str]:
    values: list[str] = []
    for part in str(raw).replace("\n", ",").split(","):
        item = part.strip()
        if item:
            values.append(item)
    return values


def read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def update_env_file(path: Path, updates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    remaining = dict(updates)
    out: list[str] = []
    for raw_line in existing_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(raw_line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            out.append(f'{key}="{_escape_env(remaining.pop(key))}"')
        else:
            out.append(raw_line)
    if remaining:
        if out and out[-1].strip():
            out.append("")
        out.append("# Site Audit local settings")
        for key in sorted(remaining):
            out.append(f'{key}="{_escape_env(remaining[key])}"')
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def _escape_env(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')
