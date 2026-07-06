"""AI crawler access checks from robots.txt and llms.txt metadata."""

from __future__ import annotations

from .robots_txt import evaluate_path, has_blanket_disallow


AI_USER_AGENTS = [
    {
        "agent": "GPTBot",
        "operator": "OpenAI",
        "purpose": "training",
        "blocking_consequence": "OpenAI cannot use this site for model training.",
    },
    {
        "agent": "OAI-SearchBot",
        "operator": "OpenAI",
        "purpose": "search",
        "blocking_consequence": "OpenAI search crawlers cannot discover or refresh this content.",
    },
    {
        "agent": "ChatGPT-User",
        "operator": "OpenAI",
        "purpose": "user_fetch",
        "blocking_consequence": "ChatGPT cannot fetch this page when a user asks it to browse or summarize the URL.",
    },
    {
        "agent": "ClaudeBot",
        "operator": "Anthropic",
        "purpose": "training",
        "blocking_consequence": "Anthropic cannot use this site for model training.",
    },
    {
        "agent": "Claude-SearchBot",
        "operator": "Anthropic",
        "purpose": "search",
        "blocking_consequence": "Claude search crawlers cannot discover or refresh this content.",
    },
    {
        "agent": "Claude-User",
        "operator": "Anthropic",
        "purpose": "user_fetch",
        "blocking_consequence": "Claude cannot fetch this page when a user asks it to browse or summarize the URL.",
    },
    {
        "agent": "PerplexityBot",
        "operator": "Perplexity",
        "purpose": "search",
        "blocking_consequence": "Perplexity cannot crawl the site for answer and search results.",
    },
    {
        "agent": "Perplexity-User",
        "operator": "Perplexity",
        "purpose": "user_fetch",
        "blocking_consequence": "Perplexity cannot fetch this page when a user asks it to open the URL.",
    },
    {
        "agent": "Google-Extended",
        "operator": "Google",
        "purpose": "training",
        "blocking_consequence": "This controls Gemini training and does not affect Google Search.",
    },
    {
        "agent": "Applebot-Extended",
        "operator": "Apple",
        "purpose": "training",
        "blocking_consequence": "Apple cannot use this site for model training.",
    },
    {
        "agent": "CCBot",
        "operator": "Common Crawl",
        "purpose": "training",
        "blocking_consequence": "Common Crawl cannot include this site in training data collections.",
    },
    {
        "agent": "Bytespider",
        "operator": "ByteDance",
        "purpose": "training",
        "blocking_consequence": "ByteDance cannot use this site for model training.",
    },
    {
        "agent": "meta-externalagent",
        "operator": "Meta",
        "purpose": "training",
        "blocking_consequence": "Meta cannot use this site for model training.",
    },
    {
        "agent": "Amazonbot",
        "operator": "Amazon",
        "purpose": "training",
        "blocking_consequence": "Amazon cannot crawl this site for AI search or training systems.",
    },
]


NO_ROBOTS_REASON = "no robots.txt found (all crawlers allowed)"
ROBOTS_UNREACHABLE_REASON = "robots.txt unreachable — access could not be evaluated"


def format_blocked_agent_recommendation(row: dict) -> str:
    """Single source of truth for the per-agent blocked-crawler sentence."""
    return (
        f"robots.txt blocks {row.get('agent')} — {row.get('blocking_consequence')} "
        "Allow it or accept invisibility in that engine."
    )


def build_ai_access(
    robots_txt: str | None,
    llms_txt: dict | None,
    llms_full_txt: dict | None,
    base_url: str,
    *,
    robots_status: int | None = None,
) -> dict:
    """Build an AI crawler access payload from fetched crawl metadata."""
    if robots_txt is None:
        status = int(robots_status or 0) if robots_status is not None else None
        # RFC 9309 section 2.3.1: 4xx means no rules (allow all), while a
        # server error / unreachable robots.txt gives no basis for claims.
        if status is not None and (status <= 0 or status >= 500):
            rows = [_unknown_agent_row(entry) for entry in AI_USER_AGENTS]
            return {
                "available": False,
                "evaluated": False,
                "base_url": base_url,
                "reason": ROBOTS_UNREACHABLE_REASON,
                "notes": [ROBOTS_UNREACHABLE_REASON],
                "agents": rows,
                "llms_txt": _llms_payload(llms_txt),
                "llms_full_txt": _llms_payload(llms_full_txt),
                "summary": _summary(rows, blanket_block=False),
                "recommendations": [],
            }
        rows = [_allowed_agent_row(entry) for entry in AI_USER_AGENTS]
        return {
            "available": False,
            "evaluated": True,
            "base_url": base_url,
            "reason": NO_ROBOTS_REASON,
            "notes": [NO_ROBOTS_REASON],
            "agents": rows,
            "llms_txt": _llms_payload(llms_txt),
            "llms_full_txt": _llms_payload(llms_full_txt),
            "summary": _summary(rows, blanket_block=False),
            "recommendations": [],
        }

    rows = []
    for entry in AI_USER_AGENTS:
        decision = evaluate_path(robots_txt, str(entry["agent"]), "/")
        rows.append({
            "agent": entry["agent"],
            "operator": entry["operator"],
            "purpose": entry["purpose"],
            "blocking_consequence": entry["blocking_consequence"],
            "allowed_root": bool(decision.get("allowed")),
            "explicitly_named": bool(decision.get("explicitly_named")),
            "matched_group": decision.get("matched_group") or "",
        })

    blanket_block = has_blanket_disallow(robots_txt) and not any(row["explicitly_named"] for row in rows)
    return {
        "available": True,
        "evaluated": True,
        "base_url": base_url,
        "reason": "",
        "notes": _notes(rows),
        "agents": rows,
        "llms_txt": _llms_payload(llms_txt),
        "llms_full_txt": _llms_payload(llms_full_txt),
        "summary": _summary(rows, blanket_block=blanket_block),
        "recommendations": _recommendations(rows, blanket_block=blanket_block),
    }


def _allowed_agent_row(entry: dict) -> dict:
    return {
        "agent": entry["agent"],
        "operator": entry["operator"],
        "purpose": entry["purpose"],
        "blocking_consequence": entry["blocking_consequence"],
        "allowed_root": True,
        "explicitly_named": False,
        "matched_group": "",
    }


def _unknown_agent_row(entry: dict) -> dict:
    row = _allowed_agent_row(entry)
    row["allowed_root"] = None
    return row


def _summary(rows: list[dict], *, blanket_block: bool) -> dict:
    # allowed_root is None when robots.txt was unreachable — count only
    # explicit False as blocked so unknown states never inflate the counts.
    return {
        "search_blocked": sum(1 for row in rows if row.get("purpose") == "search" and row.get("allowed_root") is False),
        "user_fetch_blocked": sum(1 for row in rows if row.get("purpose") == "user_fetch" and row.get("allowed_root") is False),
        "training_blocked": sum(1 for row in rows if row.get("purpose") == "training" and row.get("allowed_root") is False),
        "total_agents": len(rows),
        "blanket_block": bool(blanket_block),
    }


def _recommendations(rows: list[dict], *, blanket_block: bool) -> list[str]:
    if blanket_block:
        return [
            "Critical: robots.txt blanket-blocks AI crawlers via User-agent: * and Disallow: / — AI search and user-fetch engines cannot access the site. Add agent-specific Allow rules or accept invisibility across those engines."
        ]
    recommendations = []
    for row in rows:
        if row.get("allowed_root") is not False or row.get("purpose") == "training":
            continue
        recommendations.append(format_blocked_agent_recommendation(row))
    return recommendations


def _notes(rows: list[dict]) -> list[str]:
    out = []
    for row in rows:
        if row.get("allowed_root") is not False or row.get("purpose") != "training":
            continue
        out.append(f"robots.txt blocks training bot {row.get('agent')} ({row.get('operator')}) — {row.get('blocking_consequence')}")
    return out


def _llms_payload(payload: dict | None) -> dict:
    if not payload:
        return {
            "present": False,
            "url": "",
            "size_bytes": 0,
            "first_lines": [],
        }
    lines = [str(line) for line in (payload.get("first_lines") or [])[:20]]
    return {
        "present": bool(payload.get("present")),
        "url": str(payload.get("url") or ""),
        "size_bytes": int(payload.get("size_bytes") or 0),
        "first_lines": lines,
    }
