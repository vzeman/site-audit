"""AI-agent helpers for report-to-edit workflows.

The module keeps provider details outside the analysis pipeline. OpenRouter is
available as a direct chat-completions client. Harnext is available as a
separate coding-agent provider through ``harnext_sdk`` and the ``harnext`` CLI.
When Harnext is selected, failures are reported explicitly instead of silently
falling back to direct OpenRouter.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from .cache import content_hash
from .config_env import load_dotenv


DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-pro"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class MissingOpenRouterKey(RuntimeError):
    """Raised when an agent call is requested without an OpenRouter key."""


class AgentClient(Protocol):
    provider: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        timeout: int = 120,
    ) -> "AgentCompletion":
        ...


@dataclass
class AgentCompletion:
    text: str
    provider: str
    model: str
    cache_status: str = "miss"
    fallback_from: str = ""
    raw: dict[str, Any] | None = None


class OpenRouterClient:
    provider = "openrouter"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or openrouter_api_key()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        timeout: int = 120,
    ) -> AgentCompletion:
        if not self.api_key:
            raise MissingOpenRouterKey("Set OPENROUTER_API_KEY in .env or the environment.")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        response = requests.post(
            OPENROUTER_CHAT_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/vzeman/site-audit",
                "X-Title": "site-audit",
            },
            json=payload,
            timeout=timeout,
        )
        if response.status_code >= 400:
            message = response.text[:500]
            raise RuntimeError(f"OpenRouter request failed with HTTP {response.status_code}: {message}")
        data = response.json()
        choices = data.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            text = str(message.get("content") or "").strip()
        if not text:
            raise RuntimeError("OpenRouter returned an empty completion.")
        return AgentCompletion(text=text, provider=self.provider, model=model, raw=data)


class HarnextOpenRouterClient:
    provider = "harnext"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or openrouter_api_key()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        timeout: int = 120,
    ) -> AgentCompletion:
        if not self.api_key:
            raise MissingOpenRouterKey("Set OPENROUTER_API_KEY in .env or the environment.")
        completion = self._complete_with_harnext(messages, model=model, temperature=temperature, timeout=timeout)
        completion.provider = self.provider
        return completion

    def _complete_with_harnext(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        timeout: int,
    ) -> AgentCompletion:
        from harnext_sdk import HarnextAgentOptions, query  # type: ignore
        from harnext_sdk.types import AssistantMessage, ResultMessage, TextBlock  # type: ignore

        prompt = messages_to_prompt(messages)
        env = {
            "OPENROUTER_API_KEY": self.api_key,
            "OPENROUTER_MODEL": model,
        }
        options = HarnextAgentOptions(
            provider="openrouter",
            model=model,
            max_turns=3,
            env=env,
            auto_update_cli=True,
        )
        assistant_parts: list[str] = []
        result_text = ""
        result_payload: dict[str, Any] = {}

        async def _run() -> None:
            nonlocal result_text
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text:
                            assistant_parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    result_payload.update(
                        {
                            "subtype": message.subtype,
                            "is_error": message.is_error,
                            "result": message.result,
                            "session_id": message.session_id,
                            "num_turns": message.num_turns,
                            "duration_ms": message.duration_ms,
                            "total_cost_usd": message.total_cost_usd,
                            "usage": message.usage,
                        }
                    )
                    if message.result:
                        result_text = message.result
                    if message.is_error:
                        raise RuntimeError(message.result or "Harnext returned an error result.")

        _run_async(_run())
        text = (result_text or "\n".join(part.strip() for part in assistant_parts if part and part.strip())).strip()
        if not text:
            raise RuntimeError("Harnext returned an empty completion.")
        return AgentCompletion(text=text, provider=self.provider, model=model, raw=result_payload or {"result": text})


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Harnext agent cannot run inside an already running asyncio loop.")


def harnext_status() -> tuple[bool, str]:
    try:
        from harnext_sdk import resolve_cli_invocation  # type: ignore
    except Exception as exc:
        return False, f"Install the Python SDK with `python -m pip install harnext` or `python -m pip install -e '.[agent]'` ({exc.__class__.__name__})."
    try:
        command = resolve_cli_invocation(None)
    except Exception as exc:
        return False, f"Install the Harnext CLI with `npm install -g harnext`, or set HARNEXT_CLI_PATH ({exc})."
    return True, " ".join(command)


def openrouter_api_key() -> str:
    load_dotenv()
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def openrouter_model(default: str = DEFAULT_OPENROUTER_MODEL) -> str:
    load_dotenv()
    return os.getenv("OPENROUTER_MODEL", "").strip() or default


def build_agent_client(provider: str = "harnext", api_key: str | None = None) -> AgentClient:
    normalized = (provider or "harnext").strip().lower()
    if normalized == "openrouter":
        return OpenRouterClient(api_key=api_key)
    if normalized == "harnext":
        return HarnextOpenRouterClient(api_key=api_key)
    raise ValueError(f"Unsupported AI agent provider: {provider}")


def messages_to_prompt(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role") or "user").upper()
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def cached_completion(
    cache_dir: Path,
    *,
    kind: str,
    messages: list[dict[str, str]],
    client: AgentClient,
    model: str,
    refresh: bool = False,
    temperature: float = 0.2,
    timeout: int = 120,
) -> AgentCompletion:
    root = cache_dir / "ai_agent" / _safe_name(kind)
    root.mkdir(parents=True, exist_ok=True)
    key = content_hash(json.dumps({"kind": kind, "model": model, "messages": messages}, sort_keys=True))
    prompt_path = root / f"{key}.prompt.json"
    output_path = root / f"{key}.completion.json"
    prompt_path.write_text(
        json.dumps(
            {
                "kind": kind,
                "model": model,
                "provider": getattr(client, "provider", ""),
                "messages": messages,
                "created_at": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if output_path.is_file() and not refresh:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return AgentCompletion(
            text=str(payload.get("text") or ""),
            provider=str(payload.get("provider") or getattr(client, "provider", "")),
            model=str(payload.get("model") or model),
            cache_status="hit",
            fallback_from=str(payload.get("fallback_from") or ""),
            raw=payload.get("raw") if isinstance(payload.get("raw"), dict) else None,
        )
    completion = client.complete(messages, model=model, temperature=temperature, timeout=timeout)
    completion.cache_status = "miss"
    output_path.write_text(
        json.dumps(
            {
                "text": completion.text,
                "provider": completion.provider,
                "model": completion.model,
                "fallback_from": completion.fallback_from,
                "raw": completion.raw or {},
                "created_at": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return completion


RECOMMENDATION_SCHEMA_DOC = """## recommendation.json contract

Output a single JSON object with exactly these top-level keys:

```json
{
  "page_assessment": {"is_right_target_page": true, "reason": "state whether to retarget this page, create a new page, or proceed; address intent mismatch and winnability gates first"},
  "title": {"current": "existing title", "recommended": "new or same title", "reason": "evidence-based reason"},
  "meta_description": {"recommended": "new meta description", "reason": "reason"},
  "h1": {"recommended": "new or same H1", "reason": "reason"},
  "outline": [
    {"level": 2, "heading": "section heading", "status": "keep|rename|new|remove",
     "maps_to_topic": "topic label from evidence or empty string", "source_paragraphs": [0, 1]}
  ],
  "paragraph_decisions": [
    {"index": 0, "decision": "keep|rewrite|move|merge|remove",
     "reason": "short reason", "rewrite": "full replacement text when decision is rewrite, otherwise null"}
  ],
  "new_sections": [
    {"heading": "new section heading", "placement_after_paragraph": 7, "topic": "topic label",
     "format": "paragraphs|table|faq|steps", "draft": "full original draft copy",
     "covers_paa": ["question covered by this section"]}
  ],
  "structured_data": [{"type": "FAQPage", "reason": "why"}],
  "internal_links": [{"anchor": "anchor text", "from_hint": "what kind of page should link here", "reason": "why"}]
}
```

Rules: every paragraph index of the page appears exactly once in `paragraph_decisions`;
`page_assessment.reason` must explicitly say whether the current page should be retargeted, a new page should be created,
or editing should proceed, especially when evidence contains an intent mismatch or unlikely winnability band;
`placement_after_paragraph` is -1 for the top of the page or a valid paragraph index;
`rewrite` must be non-empty exactly when decision is `rewrite`; every new section needs a non-empty `draft`;
`title.recommended` must be at most 65 characters and `meta_description.recommended` at most 165 characters
(longer values are truncated in Google SERPs and will fail validation)."""

_DECISION_ENUM = {"keep", "rewrite", "move", "merge", "remove"}
_OUTLINE_STATUS_ENUM = {"keep", "rename", "new", "remove"}
_SECTION_FORMAT_ENUM = {"paragraphs", "table", "faq", "steps"}


def parse_recommendation(text: str) -> dict:
    payload = _extract_json(text)
    if isinstance(payload, list) and payload:
        payload = payload[0]
    return payload if isinstance(payload, dict) else {}


def validate_recommendation(payload: dict, paragraph_count: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict) or not payload:
        return ["recommendation is empty or not a JSON object"]
    for key, expected in (
        ("page_assessment", dict),
        ("title", dict),
        ("meta_description", dict),
        ("h1", dict),
        ("outline", list),
        ("paragraph_decisions", list),
        ("new_sections", list),
        ("structured_data", list),
        ("internal_links", list),
    ):
        if key not in payload:
            errors.append(f"missing key: {key}")
        elif not isinstance(payload.get(key), expected):
            errors.append(f"key {key} must be {expected.__name__}")
    recommended_title = str((payload.get("title") or {}).get("recommended") or "")
    if len(recommended_title) > 65:
        errors.append(f"title.recommended is {len(recommended_title)} characters; maximum is 65 (SERP truncation)")
    recommended_meta = str((payload.get("meta_description") or {}).get("recommended") or "")
    if len(recommended_meta) > 165:
        errors.append(f"meta_description.recommended is {len(recommended_meta)} characters; maximum is 165 (SERP truncation)")
    decisions = payload.get("paragraph_decisions") or []
    if isinstance(decisions, list):
        seen: set[int] = set()
        for i, row in enumerate(decisions):
            if not isinstance(row, dict):
                errors.append(f"paragraph_decisions[{i}] is not an object")
                continue
            index = row.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < max(paragraph_count, 1)):
                errors.append(f"paragraph_decisions[{i}].index invalid: {index!r}")
            elif index in seen:
                errors.append(f"paragraph_decisions has duplicate index {index}")
            else:
                seen.add(index)
            decision = str(row.get("decision") or "")
            if decision not in _DECISION_ENUM:
                errors.append(f"paragraph_decisions[{i}].decision invalid: {decision!r}")
            rewrite = row.get("rewrite")
            if decision == "rewrite" and not (isinstance(rewrite, str) and rewrite.strip()):
                errors.append(f"paragraph_decisions[{i}] decision is rewrite but rewrite text is empty")
            if decision != "rewrite" and isinstance(rewrite, str) and rewrite.strip():
                errors.append(f"paragraph_decisions[{i}] has rewrite text but decision is {decision!r}")
        if paragraph_count > 0:
            missing = sorted(set(range(paragraph_count)) - seen)
            if missing:
                preview = ", ".join(str(x) for x in missing[:10])
                errors.append(f"paragraph_decisions missing indexes: {preview}" + (" …" if len(missing) > 10 else ""))
    for i, row in enumerate(payload.get("outline") or []):
        if not isinstance(row, dict):
            errors.append(f"outline[{i}] is not an object")
            continue
        if not str(row.get("heading") or "").strip():
            errors.append(f"outline[{i}].heading is empty")
        if str(row.get("status") or "") not in _OUTLINE_STATUS_ENUM:
            errors.append(f"outline[{i}].status invalid: {row.get('status')!r}")
    for i, row in enumerate(payload.get("new_sections") or []):
        if not isinstance(row, dict):
            errors.append(f"new_sections[{i}] is not an object")
            continue
        if not str(row.get("heading") or "").strip():
            errors.append(f"new_sections[{i}].heading is empty")
        if not str(row.get("draft") or "").strip():
            errors.append(f"new_sections[{i}].draft is empty")
        fmt = str(row.get("format") or "")
        if fmt and fmt not in _SECTION_FORMAT_ENUM:
            errors.append(f"new_sections[{i}].format invalid: {fmt!r}")
        placement = row.get("placement_after_paragraph")
        if placement is not None and (
            not isinstance(placement, int) or isinstance(placement, bool)
            or not (-1 <= placement < max(paragraph_count, 1))
        ):
            errors.append(f"new_sections[{i}].placement_after_paragraph invalid: {placement!r}")
    has_removals = any(
        isinstance(row, dict) and row.get("decision") == "remove" for row in decisions if isinstance(decisions, list)
    )
    if (has_removals or (payload.get("new_sections") or [])) and not (payload.get("outline") or []):
        errors.append("outline must be provided when paragraphs are removed or new sections are added")
    return errors


def _workspace_session_prompt(workspace: Path) -> str:
    return (
        "You are a senior SEO/GEO editor working inside the directory "
        f"{workspace}. Read TASK.md first, then evidence.json and our_page.md. "
        "Consult the competitors/ directory and serp.json as needed. "
        "Write two files into this directory: recommendation.json (must satisfy the contract in TASK.md exactly) "
        "and brief.md (human-readable editorial brief). Do not modify any other file."
    )


def _harnext_session_runner(
    prompt: str,
    *,
    workspace: Path,
    model: str,
    max_turns: int,
    api_key: str | None = None,
) -> dict[str, Any]:
    from harnext_sdk import HarnextAgentOptions, query  # type: ignore
    from harnext_sdk.types import AssistantMessage, ResultMessage, TextBlock  # type: ignore
    import inspect

    key = api_key or openrouter_api_key()
    if not key:
        raise MissingOpenRouterKey("Set OPENROUTER_API_KEY in .env or the environment.")
    kwargs: dict[str, Any] = {
        "provider": "openrouter",
        "model": model,
        "max_turns": max_turns,
        "env": {"OPENROUTER_API_KEY": key, "OPENROUTER_MODEL": model},
        "auto_update_cli": True,
    }
    try:
        params = set(inspect.signature(HarnextAgentOptions).parameters)
    except (TypeError, ValueError):
        params = set()
    for candidate in ("cwd", "workdir", "working_directory"):
        if candidate in params:
            kwargs[candidate] = str(workspace)
            break
    if "permission_mode" in params:
        kwargs["permission_mode"] = "acceptEdits"
    options = HarnextAgentOptions(**kwargs)
    assistant_parts: list[str] = []
    result_payload: dict[str, Any] = {}

    async def _run() -> None:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        assistant_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                result_payload.update({
                    "subtype": message.subtype,
                    "is_error": message.is_error,
                    "result": message.result,
                    "session_id": message.session_id,
                    "num_turns": message.num_turns,
                    "duration_ms": message.duration_ms,
                    "total_cost_usd": message.total_cost_usd,
                    "usage": message.usage,
                })
                if message.is_error:
                    raise RuntimeError(message.result or "Harnext returned an error result.")

    _run_async(_run())
    result_payload.setdefault("assistant_text", "\n".join(p.strip() for p in assistant_parts if p.strip()))
    return result_payload


def run_harnext_workspace_session(
    workspace: Path,
    *,
    model: str,
    max_turns: int = 20,
    timeout: int = 600,
    api_key: str | None = None,
    session_runner: Any = None,
    extra_prompt: str = "",
) -> AgentCompletion:
    workspace = Path(workspace)
    prompt = _workspace_session_prompt(workspace)
    if extra_prompt.strip():
        prompt = f"{prompt}\n\n{extra_prompt.strip()}"
    runner = session_runner or _harnext_session_runner
    raw_session: dict[str, Any] = {}
    session_error = ""
    try:
        raw_session = runner(prompt, workspace=workspace, model=model, max_turns=max_turns, api_key=api_key) or {}
    except MissingOpenRouterKey:
        raise
    except Exception as exc:
        # Harnext can return an error result after successfully writing the
        # requested workspace files. Prefer the files when they exist, while
        # preserving the session error for diagnostics.
        session_error = str(exc)
        raw_session = {"is_error": True, "result": session_error}
    recommendation: dict = {}
    rec_path = workspace / "recommendation.json"
    if rec_path.is_file():
        try:
            recommendation = json.loads(rec_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            recommendation = parse_recommendation(rec_path.read_text(encoding="utf-8", errors="replace"))
    brief = ""
    brief_path = workspace / "brief.md"
    if brief_path.is_file():
        try:
            brief = brief_path.read_text(encoding="utf-8").strip()
        except OSError:
            brief = ""
    if not brief:
        brief = str(raw_session.get("result") or raw_session.get("assistant_text") or "").strip()
    if not brief and not recommendation:
        raise RuntimeError(session_error or "Harnext workspace session produced neither brief.md nor recommendation.json.")
    if session_error:
        raw_session.setdefault("file_output_after_error", True)
    return AgentCompletion(
        text=brief,
        provider="harnext",
        model=model,
        raw={"recommendation": recommendation, "session": raw_session},
    )


def cached_workspace_completion(
    cache_dir: Path,
    *,
    kind: str,
    key: str,
    runner: Any,
    refresh: bool = False,
) -> dict:
    root = cache_dir / "ai_agent" / _safe_name(kind)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{key}.workspace.json"
    if output_path.is_file() and not refresh:
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            payload["cache_status"] = "hit"
            return payload
        except (json.JSONDecodeError, OSError):
            pass
    payload = runner() or {}
    payload["cache_status"] = "miss"
    payload.setdefault("created_at", time.time())
    if not payload.get("errors"):
        # Only cache clean results; a failed recommendation should be retried
        # on the next run instead of being frozen by the cache.
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_keyword_messages(page: dict, *, max_keywords: int = 5) -> list[dict[str, str]]:
    page_payload = {
        "url": page.get("url", ""),
        "title": page.get("title", ""),
        "h1": page.get("h1", ""),
        "description": page.get("description", ""),
        "section": page.get("section", ""),
        "headers": (page.get("headers") or [])[:20],
        "paragraphs": (page.get("paragraphs") or [])[:12],
        "known_search_rows": (page.get("search_rows") or [])[:20],
    }
    return [
        {
            "role": "system",
            "content": (
                "You select search keywords for a single URL before SERP gap analysis. "
                "Use only evidence from the page and supplied search rows. Prefer phrases with clear intent. "
                "Do not invent demand metrics. Return compact JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Pick up to {max_keywords} target keywords for this URL. "
                "Prioritize terms the page can realistically rank for and that match the main intent. "
                "Return exactly this JSON shape: "
                '{"keywords":[{"keyword":"phrase","intent":"why this matches","priority":1}]}\n\n'
                f"Evidence:\n{json.dumps(page_payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def build_language_detection_messages(evidence: dict) -> list[dict[str, str]]:
    page_payload = {
        "pages": (evidence.get("pages") or [])[:5],
        "existing_language_codes": (evidence.get("existing_language_codes") or [])[:8],
        "search_rows": (evidence.get("search_rows") or [])[:20],
    }
    return [
        {
            "role": "system",
            "content": (
                "You detect the dominant natural language for SERP gap analysis. "
                "Use only supplied page evidence. Prefer the language readers see in the main content, "
                "not boilerplate, brand names, URLs, or code. Return compact JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Detect the best Google SERP language code for these selected page(s). "
                "Return exactly this JSON shape: "
                '{"language_code":"en","language_name":"English","confidence":0.0,"reason":"short evidence"}\n\n'
                "Use a lowercase ISO-style language code suitable for Google SERP APIs, for example en, sk, cs, de, es, fr, pl. "
                "If evidence is mixed, choose the dominant main-content language and explain briefly.\n\n"
                f"Evidence:\n{json.dumps(page_payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def parse_language_detection(text: str) -> dict[str, Any]:
    payload = _extract_json(text)
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return {}
    code = _normalize_language_code(
        payload.get("language_code")
        or payload.get("language")
        or payload.get("code")
        or payload.get("hl")
    )
    if not code:
        return {}
    return {
        "language_code": code,
        "language_name": str(payload.get("language_name") or payload.get("name") or "").strip(),
        "confidence": _safe_confidence(payload.get("confidence")),
        "reason": str(payload.get("reason") or payload.get("evidence") or "").strip(),
    }


def parse_keyword_candidates(text: str, *, limit: int = 5) -> list[str]:
    candidates: list[str] = []
    payload = _extract_json(text)
    if isinstance(payload, dict):
        raw_items = payload.get("keywords") or payload.get("queries") or payload.get("terms") or []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []
    for item in raw_items:
        if isinstance(item, dict):
            value = item.get("keyword") or item.get("query") or item.get("term")
        else:
            value = item
        _append_keyword_candidate(candidates, str(value or ""), limit)
    if not candidates:
        for line in str(text or "").splitlines():
            stripped = re.sub(r"^\s*[-*0-9.)]+\s*", "", line).strip()
            if stripped.startswith("```") or stripped in {"{", "}", "[", "]", "},", "],"}:
                continue
            if ":" in stripped:
                stripped = stripped.split(":", 1)[-1].strip()
            _append_keyword_candidate(candidates, stripped, limit)
            if len(candidates) >= limit:
                break
    return candidates[:limit]


def fallback_keyword_candidates(page: dict, *, limit: int = 5) -> list[str]:
    values = [
        page.get("h1"),
        page.get("title"),
        page.get("section"),
    ]
    values.extend(page.get("headers") or [])
    out: list[str] = []
    for value in values:
        text = re.sub(r"\s+[-|:]\s+.*$", "", str(value or "")).strip()
        text = re.sub(r"\s+", " ", text)
        _append_keyword_candidate(out, text, limit)
        if len(out) >= limit:
            break
    return out[:limit]


def build_editor_brief_messages(page: dict) -> list[dict[str, str]]:
    payload = _editor_prompt_payload(page)
    return [
        {
            "role": "system",
            "content": (
                "You are a senior SEO editor writing implementation instructions for one analyzed URL. "
                "Be precise, evidence-led, and URL-specific. No generic SEO advice, no ballast, no duplicate tasks, "
                "and no copied competitor wording. Do not copy competitor wording. Separate evidence, actions, draft copy, "
                "and a final article draft that can be handed to an implementation agent. Use only the Evidence JSON supplied "
                "in the user message. Do not inspect files, browse, run commands, or ask for more context."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create a markdown TODO brief for the Harnext AI coding/content agent. "
                "Do not use tools and do not read files; all required evidence is below. "
                "Use this exact structure and keep each bullet actionable:\n"
                "# AI Agent TODO\n"
                "## Evidence\n"
                "## Paragraph Decisions\n"
                "## Sections To Add Or Expand\n"
                "## Recommended Content Order\n"
                "## Draft Copy\n"
                "## Final Article Draft\n"
                "## Acceptance Criteria\n\n"
                "Our page's complete content is in own_page (headings in order, paragraphs numbered by index). "
                "In Paragraph Decisions, give a decision (keep, rewrite, move, merge, or remove) for every paragraph listed "
                "in paragraph_review, referencing paragraphs as [P<index>]. For paragraphs not listed, only mention them if "
                "they conflict with a new section. "
                "How to read similarity scores: >= 0.78 covered, 0.62-0.78 partial, < 0.62 weak; paragraph_review is sorted "
                "weakest-first so its values can still be high — never call a score above 0.78 'low'; cite the actual number "
                "and the correct band in reasons. "
                "For missing competitor-covered topics, write the actual original draft copy that should be added. "
                "In Final Article Draft, assemble the full recommended article from own_page order: reuse kept paragraphs by "
                "reference ([P<index>] plus the first 6 words), include rewritten and new paragraphs in full, and mark "
                "remove/merge omissions explicitly. Use benchmark (median competitor paragraphs/headings) as the size target. "
                "Cover every missing-status question from serp_features.people_also_ask that matches this page's intent, "
                "either in a section or a FAQ block; mark off-intent questions as ignored (off-intent) with one line of "
                "reasoning instead of forcing coverage. PAA questions are things users ask, not facts — never turn a "
                "question's wording into a claim. "
                "If an analysis has intent.match = mismatch, page_assessment must address it first and state whether to "
                "retarget this page, create a new page, or proceed. If winnability.band = unlikely, say content changes "
                "alone are unlikely to reach page 1, recommend the supplied alternative keyword when present, and list "
                "link acquisition as the prerequisite. "
                "Ignore navigation, footer, cookie, newsletter, and language-switcher items if they appear in own_page "
                "headings. Keep the recommended title at most 65 characters and the meta description at most 165 characters. "
                "Respect structural_patterns advice (tables, question-form headings, statistics, schema). "
                "Do not duplicate topics listed in covered_topics. "
                "If impressions, clicks, traffic, or volume are absent, say demand metrics absent instead of guessing. "
                "NEVER invent statistics, percentages, time savings, or product capabilities (plans, languages, limits) "
                "that are not present in the Evidence JSON; state such claims qualitatively or mark them [NEEDS DATA]. "
                "Do not reuse a heading for two different sections; merge overlapping sections instead. "
                "Do not repeat the same instruction in multiple sections.\n\n"
                "After the markdown brief, additionally output one fenced ```json code block containing the "
                "machine-readable recommendation that follows this contract exactly:\n"
                f"{RECOMMENDATION_SCHEMA_DOC}\n\n"
                f"Evidence JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _pick(d: dict, keys: list[str]) -> dict:
    source = d if isinstance(d, dict) else {}
    return {key: source.get(key) for key in keys if key in source}


def _best_cell(row: dict) -> dict:
    best: dict = {}
    best_similarity = -1.0
    for cell in row.get("cells") or []:
        try:
            similarity = float(cell.get("similarity") or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        if similarity > best_similarity:
            best_similarity = similarity
            best = cell
    if not best:
        return {}
    return {
        "url": best.get("url", ""),
        "rank": best.get("rank"),
        "similarity": best.get("similarity"),
        "paragraph": str(best.get("paragraph") or "")[:320],
    }


def _editor_prompt_payload(page: dict) -> dict:
    actions = sorted(page.get("action_points") or [], key=lambda row: -float(row.get("impact_score") or 0))[:18]
    analyses = []
    for analysis in page.get("analyses") or []:
        keyword = analysis.get("keyword") or {}
        topics = [
            {
                "label": topic.get("label"),
                "coverage": topic.get("coverage"),
                "priority": topic.get("priority"),
                "competitor_coverage": topic.get("competitor_coverage"),
                "our_best_similarity": topic.get("our_best_similarity"),
                "our_best_paragraph_index": topic.get("our_best_paragraph_index"),
                "example_url": ((topic.get("examples") or [{}])[0]).get("url", ""),
                "example_paragraph": ((topic.get("examples") or [{}])[0]).get("paragraph", "")[:320],
            }
            for topic in (analysis.get("topics") or [])[:12]
        ]
        heatmap_rows = list((analysis.get("paragraph_match_heatmap") or {}).get("rows") or [])
        heatmap_rows.sort(key=lambda row: float(row.get("max_similarity") or 0.0))
        paragraph_rows = []
        for row in heatmap_rows[:25]:
            paragraph_rows.append({
                "paragraph_index": row.get("paragraph_index"),
                "status": row.get("status"),
                "max_similarity": row.get("max_similarity"),
                "max_rank_impact": row.get("max_rank_impact"),
                "word_count": row.get("word_count"),
                "paragraph": str(row.get("paragraph") or "")[:400],
                "best_competitor": _best_cell(row),
            })
        content_order_path = analysis.get("content_order_path") or {}
        analyses.append({
            "keyword": keyword.get("keyword") or analysis.get("query", ""),
            "keyword_metrics": {
                "source": keyword.get("source", ""),
                "position": keyword.get("position", 0),
                "impressions": keyword.get("impressions", 0),
                "clicks": keyword.get("clicks", 0),
                "traffic": keyword.get("traffic", 0),
                "volume": keyword.get("volume", 0),
                "metrics_source": keyword.get("metrics_source", ""),
            },
            "summary": analysis.get("summary") or {},
            "intent": analysis.get("intent") or {},
            "winnability": analysis.get("winnability") or {},
            "alternative_keyword": analysis.get("alternative_keyword") or {},
            "recommendation_header": analysis.get("recommendation_header") or "",
            "benchmark": (analysis.get("content_comparison") or {}).get("benchmark") or {},
            "our_profile": _pick(
                (analysis.get("content_comparison") or {}).get("ours") or {},
                ["paragraph_count", "word_count", "heading_count", "h2_h3_count", "coverage_ratio"],
            ),
            "structural_patterns": (analysis.get("structural_patterns") or [])[:8],
            "serp_features": {
                "people_also_ask": [
                    {
                        "question": row.get("question"),
                        "status": row.get("status"),
                        "best_similarity": row.get("best_similarity"),
                    }
                    for row in (analysis.get("paa_coverage") or [])[:10]
                ],
                "related_searches": (analysis.get("serp_features") or {}).get("related_searches") or [],
                "answer_box": (analysis.get("serp_features") or {}).get("answer_box") or {},
                "ai_overview": (analysis.get("serp_features") or {}).get("ai_overview"),
            },
            "topics": topics,
            "covered_topics": [
                str(topic.get("label") or "") for topic in analysis.get("covered_topics") or []
            ][:12],
            "content_order": {
                "summary": content_order_path.get("summary") or {},
                "missing_clusters": [
                    {
                        "label": cluster.get("label"),
                        "competitor_pages": cluster.get("competitor_pages"),
                        "sample_text": str(cluster.get("sample_text") or "")[:200],
                    }
                    for cluster in content_order_path.get("missing_clusters") or []
                ][:8],
                "order_deviations": [
                    {
                        "label": cluster.get("label"),
                        "direction": cluster.get("direction"),
                        "delta": cluster.get("delta"),
                    }
                    for cluster in content_order_path.get("deviations") or []
                ][:8],
            },
            "paragraph_review": paragraph_rows,
        })
    payload = {
        "url": page.get("url", ""),
        "title": page.get("title", ""),
        "h1": page.get("h1", ""),
        "keywords": page.get("keywords") or [],
        "own_page": page.get("own_content") or {},
        "content_brief": page.get("content_brief") or {},
        "actions": [
            {
                "priority": action.get("priority", ""),
                "type": action.get("type", ""),
                "keyword": action.get("keyword", ""),
                "topic": action.get("topic", ""),
                "task_summary": action.get("task_summary", ""),
                "instruction": action.get("instruction", ""),
                "placement": action.get("placement", ""),
                "acceptance_criteria": action.get("acceptance_criteria") or [],
                "evidence": action.get("evidence") or {},
            }
            for action in actions
        ],
        "analyses": analyses,
    }
    return _shrink_editor_payload(payload)


def _shrink_editor_payload(payload: dict, max_chars: int = 120_000) -> dict:
    def size() -> int:
        return len(json.dumps(payload, ensure_ascii=False))

    if size() <= max_chars:
        return payload
    for paragraph in (payload.get("own_page") or {}).get("paragraphs") or []:
        paragraph["text"] = str(paragraph.get("text") or "")[:240]
    if size() <= max_chars:
        return payload
    for analysis in payload.get("analyses") or []:
        analysis["paragraph_review"] = (analysis.get("paragraph_review") or [])[:15]
    if size() <= max_chars:
        return payload
    for analysis in payload.get("analyses") or []:
        for topic in analysis.get("topics") or []:
            topic["example_paragraph"] = str(topic.get("example_paragraph") or "")[:160]
        for row in analysis.get("paragraph_review") or []:
            best = row.get("best_competitor") or {}
            if best.get("paragraph"):
                best["paragraph"] = str(best["paragraph"])[:160]
    if size() <= max_chars:
        return payload
    for analysis in payload.get("analyses") or []:
        for cluster in (analysis.get("content_order") or {}).get("missing_clusters") or []:
            cluster["sample_text"] = ""
    if size() <= max_chars:
        return payload

    def truncate_strings(node, limit: int):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and len(value) > limit:
                    node[key] = value[:limit]
                else:
                    truncate_strings(value, limit)
        elif isinstance(node, list):
            for item in node:
                truncate_strings(item, limit)

    for limit in (160, 80, 40):
        truncate_strings(payload, limit)
        if size() <= max_chars:
            return payload
    return payload


def _extract_json(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [raw, *fenced, _slice_between(raw, "{", "}"), _slice_between(raw, "[", "]")]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
        except TypeError:
            continue
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(raw[index:])
            return payload
        except json.JSONDecodeError:
            continue
    return None


def _slice_between(text: str, start: str, end: str) -> str:
    left = text.find(start)
    right = text.rfind(end)
    if left < 0 or right < left:
        return ""
    return text[left:right + 1]


def _append_keyword_candidate(out: list[str], value: str, limit: int) -> None:
    normalized = re.sub(r"\s+", " ", value).strip().strip("\"'`.,;:")
    normalized = re.sub(r"\s+-\s+.*$", "", normalized).strip()
    if not re.search(r"[A-Za-z0-9]", normalized):
        return
    if not normalized or "://" in normalized or len(normalized) > 90:
        return
    if len(normalized.split()) > 9:
        return
    key = normalized.lower()
    if key in {item.lower() for item in out}:
        return
    out.append(normalized)
    if len(out) > limit:
        del out[limit:]


def _normalize_language_code(value: Any) -> str:
    code = re.sub(r"[^a-zA-Z0-9_-]+", "", str(value or "").strip()).replace("_", "-").lower()
    if not code:
        return ""
    if "-" in code:
        parts = [part for part in code.split("-") if part]
        if not parts:
            return ""
        code = parts[0]
    if not re.fullmatch(r"[a-z]{2,3}", code):
        return ""
    return code


def _safe_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _completion_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    for attr in ("text", "output", "content", "message"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(result, dict):
        for key in ("text", "output", "content", "message"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(result or "").strip()


def _safe_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value or "").strip().lower()).strip("-")
    return slug or "completion"
