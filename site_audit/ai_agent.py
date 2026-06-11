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
                "For existing paragraphs, say keep, rewrite, move, merge, or remove. "
                "For missing competitor-covered topics, write the actual original draft copy that should be added. "
                "In Final Article Draft, assemble the full recommended article in final reading order, including headings, "
                "replacement paragraphs, new paragraphs, and clear remove/merge omissions. "
                "If impressions, clicks, traffic, or volume are absent, say demand metrics absent instead of guessing. "
                "Do not repeat the same instruction in multiple sections.\n\n"
                f"Evidence JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


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
                "example_url": ((topic.get("examples") or [{}])[0]).get("url", ""),
                "example_paragraph": ((topic.get("examples") or [{}])[0]).get("paragraph", "")[:320],
            }
            for topic in (analysis.get("topics") or [])[:12]
        ]
        paragraph_rows = []
        for row in (analysis.get("paragraph_match_heatmap") or {}).get("rows") or []:
            paragraph_rows.append({
                "paragraph_index": row.get("paragraph_index"),
                "best_similarity": row.get("max_similarity"),
                "paragraph": str(row.get("paragraph") or "")[:320],
                "best_competitor_url": row.get("best_competitor_url", ""),
                "best_competitor_paragraph": str(row.get("best_competitor_paragraph") or "")[:320],
            })
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
            "visual_summary": analysis.get("visual_summary") or [],
            "summary": analysis.get("summary") or {},
            "topics": topics,
            "content_order_path": (analysis.get("content_order_path") or {}).get("summary", {}),
            "paragraph_review": paragraph_rows[:10],
        })
    return {
        "url": page.get("url", ""),
        "title": page.get("title", ""),
        "h1": page.get("h1", ""),
        "keywords": page.get("keywords") or [],
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
