"""robots.txt analysis helpers for technical SEO issues."""

from __future__ import annotations


def analyze(
    robots_url: str,
    status: int,
    body: str,
    *,
    final_url: str = "",
    error: str = "",
) -> dict:
    syntax_errors = _syntax_errors(body) if int(status or 0) == 200 else []
    issues = []
    if syntax_errors:
        issues.append("robots_txt_has_syntax_error")
    return {
        "url": robots_url,
        "final_url": final_url or robots_url,
        "status": int(status or 0),
        "error": error,
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
