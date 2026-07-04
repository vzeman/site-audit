"""robots.txt analysis helpers for technical SEO issues."""

from __future__ import annotations


MAX_ROBOTS_REDIRECTS = 5


def analyze(
    robots_url: str,
    status: int,
    body: str,
    *,
    final_url: str = "",
    error: str = "",
    redirect_status_codes: list[int] | None = None,
) -> dict:
    redirects = [_safe_int(code) for code in (redirect_status_codes or [])]
    syntax_errors = _syntax_errors(body) if int(status or 0) == 200 else []
    issues = []
    if syntax_errors:
        issues.append("robots_txt_has_syntax_error")
    if _has_redirect_loop_or_too_many_redirects(error, redirects):
        issues.append("robots_txt_has_too_many_redirects_or_redirect_loop")
    if _is_not_accessible(status):
        issues.append("robots_txt_is_not_accessible")
    return {
        "url": robots_url,
        "final_url": final_url or robots_url,
        "status": int(status or 0),
        "error": error,
        "redirect_status_codes": redirects,
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


def _safe_int(value) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0
