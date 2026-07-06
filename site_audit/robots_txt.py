"""robots.txt analysis helpers for technical SEO issues."""

from __future__ import annotations

import hashlib
import re


MAX_ROBOTS_REDIRECTS = 5


def parse_groups(body: str) -> list[dict]:
    """Parse robots.txt into user-agent groups with allow/disallow rules."""
    groups: list[dict] = []
    current_agents: list[str] = []
    current_rules: list[dict] = []
    seen_rule = False

    def flush() -> None:
        nonlocal current_agents, current_rules, seen_rule
        if current_agents:
            groups.append({
                "user_agents": current_agents,
                "rules": current_rules,
            })
        current_agents = []
        current_rules = []
        seen_rule = False

    for line_no, raw_line in enumerate((body or "").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        directive, value = line.split(":", 1)
        key = directive.strip().lower()
        val = value.strip()
        if key == "user-agent":
            if seen_rule:
                flush()
            if val:
                current_agents.append(val)
            continue
        if key not in {"allow", "disallow"}:
            continue
        if not current_agents:
            continue
        seen_rule = True
        current_rules.append({
            "directive": key,
            "path": val,
            "line": line_no,
        })
    flush()
    return groups


def evaluate_path(body: str, user_agent: str, path: str = "/") -> dict:
    """Evaluate a path for a user-agent using RFC 9309 group precedence."""
    groups = parse_groups(body)
    match = _best_group(groups, user_agent)
    if match is None:
        return {
            "allowed": True,
            "explicitly_named": False,
            "matched_group": "",
            "matched_rule": None,
        }
    token, _group = match
    # RFC 9309 section 2.2.1: rules from all groups sharing the matched
    # user-agent token are combined before path selection.
    rule = _best_rule(_rules_for_token(groups, token), path)
    if rule is None:
        return {
            "allowed": True,
            "explicitly_named": token != "*",
            "matched_group": token,
            "matched_rule": None,
        }
    return {
        "allowed": rule.get("directive") != "disallow",
        "explicitly_named": token != "*",
        "matched_group": token,
        "matched_rule": rule,
    }


_BLANKET_DISALLOW_PATHS = {"/", "/*", "*"}


def has_blanket_disallow(body: str) -> bool:
    """Return true when the wildcard group disallows the whole site."""
    groups = parse_groups(body)
    rules = _rules_for_token(groups, "*")
    if not rules:
        return False
    decision = _best_rule(rules, "/")
    return bool(
        decision
        and decision.get("directive") == "disallow"
        and str(decision.get("path")) in _BLANKET_DISALLOW_PATHS
    )


def analyze(
    robots_url: str,
    status: int,
    body: str,
    *,
    final_url: str = "",
    error: str = "",
    redirect_status_codes: list[int] | None = None,
    previous_body: str = "",
    previous_hash: str = "",
) -> dict:
    redirects = [_safe_int(code) for code in (redirect_status_codes or [])]
    current_hash = _content_hash(body)
    before_hash = previous_hash or _content_hash(previous_body)
    syntax_errors = _syntax_errors(body) if int(status or 0) == 200 else []
    issues = []
    if syntax_errors:
        issues.append("robots_txt_has_syntax_error")
    if _has_redirect_loop_or_too_many_redirects(error, redirects):
        issues.append("robots_txt_has_too_many_redirects_or_redirect_loop")
    if _is_not_accessible(status):
        issues.append("robots_txt_is_not_accessible")
    if before_hash and current_hash and before_hash != current_hash:
        issues.append("robots_txt_changed")
    return {
        "url": robots_url,
        "final_url": final_url or robots_url,
        "status": int(status or 0),
        "error": error,
        "redirect_status_codes": redirects,
        "content_hash": current_hash,
        "previous_content_hash": before_hash,
        "issues": issues,
        "syntax_errors": syntax_errors,
        "size_bytes": len((body or "").encode("utf-8")),
    }


def _syntax_errors(body: str) -> list[dict]:
    errors: list[dict] = []
    for line_no, raw_line in enumerate((body or "").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            errors.append({"line": line_no, "message": "Missing ':' separator", "content": raw_line[:160]})
            continue
        directive, value = line.split(":", 1)
        if not directive.strip():
            errors.append({"line": line_no, "message": "Missing directive name", "content": raw_line[:160]})
            continue
        if directive.strip().lower() in {"user-agent", "sitemap"} and not value.strip():
            errors.append({"line": line_no, "message": "Missing directive value", "content": raw_line[:160]})
    return errors


def _has_redirect_loop_or_too_many_redirects(error: str, redirect_status_codes: list[int]) -> bool:
    normalized_error = str(error or "").lower().replace("_", " ")
    if "redirect loop" in normalized_error or "too many redirect" in normalized_error:
        return True
    return len(redirect_status_codes) > MAX_ROBOTS_REDIRECTS


def _is_not_accessible(status: int) -> bool:
    status_int = _safe_int(status)
    return status_int <= 0 or status_int >= 400


def _content_hash(body: str) -> str:
    text = str(body or "").strip()
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _safe_int(value) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _best_group(groups: list[dict], user_agent: str) -> tuple[str, dict] | None:
    agent = str(user_agent or "").lower()
    best: tuple[int, int, str, dict] | None = None
    for order, group in enumerate(groups):
        for token in group.get("user_agents") or []:
            normalized = str(token or "").strip().lower()
            if not normalized:
                continue
            if normalized == "*" or normalized in agent:
                score = len(normalized) if normalized != "*" else 0
                if best is None or score > best[0] or (score == best[0] and order < best[1]):
                    best = (score, order, str(token).strip(), group)
    if best is None:
        return None
    return best[2], best[3]


def _rules_for_token(groups: list[dict], token: str) -> list[dict]:
    """Combine rules from every group naming ``token`` (RFC 9309 section 2.2.1)."""
    normalized = str(token or "").strip().lower()
    rules: list[dict] = []
    for group in groups:
        agents = group.get("user_agents") or []
        if any(str(agent or "").strip().lower() == normalized for agent in agents):
            rules.extend(group.get("rules") or [])
    return rules


def _best_rule(rules: list[dict], path: str) -> dict | None:
    best: tuple[int, int, int, dict] | None = None
    target = path or "/"
    for order, rule in enumerate(rules):
        pattern = str(rule.get("path") or "")
        if rule.get("directive") == "disallow" and pattern == "":
            continue
        if not _path_matches(pattern, target):
            continue
        # For equal-length matches, Allow wins over Disallow.
        allow_rank = 1 if rule.get("directive") == "allow" else 0
        score = len(pattern)
        if best is None or score > best[0] or (score == best[0] and allow_rank > best[1]):
            best = (score, allow_rank, order, rule)
    return best[3] if best else None


def _path_matches(pattern: str, path: str) -> bool:
    if pattern == "":
        return False
    if "*" not in pattern and "$" not in pattern:
        return path.startswith(pattern)
    anchored = pattern.endswith("$")
    raw = pattern[:-1] if anchored else pattern
    regex = "^" + ".*".join(re.escape(part) for part in raw.split("*"))
    if anchored:
        regex += "$"
    return re.match(regex, path) is not None
