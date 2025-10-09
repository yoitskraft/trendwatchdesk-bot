#!/usr/bin/env python3
import os, sys, json, datetime, subprocess, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "OPERATIONS_GUIDE.md"
README = ROOT / "README.md"

def _env(k, default=""):
    return os.getenv(k, default)

def stamp_block():
    now_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # capture key knobs if present (absent ones will be blank)
    mode = _env("TWD_MODE")
    tf = _env("TWD_TF")
    recency = _env("TWD_BREAKING_RECENCY_MIN")
    min_sources = _env("TWD_BREAKING_MIN_SOURCES")
    fallback = _env("TWD_BREAKING_FALLBACK")
    allow_rss = _env("TWD_ALLOW_RSS")
    wl = _env("TWD_WATCHLIST")
    charts_branch = "charts"
    posters_branch = "posters"

    lines = []
    lines.append("<!-- TWD_STATUS:BEGIN -->")
    lines.append("")
    lines.append("## Automation Status (auto-generated)")
    lines.append(f"- **Last run:** {now_utc}")
    if mode:
        lines.append(f"- **Mode:** `{mode}`   ·  **Timeframe:** `{tf or 'D'}`")
    if recency or min_sources or fallback:
        bits = []
        if recency: bits.append(f"recency={recency}m")
        if min_sources: bits.append(f"min_sources={min_sources}")
        if fallback: bits.append(f"fallback={fallback}")
        if allow_rss: bits.append(f"rss={allow_rss}")
        lines.append(f"- **Breaking-posters knobs:** {', '.join(bits)}")
    if wl:
        wl_fmt = ", ".join([s.strip() for s in wl.split(",") if s.strip()][:20])
        lines.append(f"- **Watchlist (top):** {wl_fmt}{' …' if len(wl_fmt) > 0 and len(wl.split(','))>20 else ''}")
    lines.append(f"- **Publish targets:** charts → `{charts_branch}`, posters → `{posters_branch}`")
    lines.append("")
    lines.append("<!-- TWD_STATUS:END -->")
    lines.append("")
    return "\n".join(lines)

def upsert_block(file_path: Path, block: str):
    content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    if "<!-- TWD_STATUS:BEGIN -->" in content and "<!-- TWD_STATUS:END -->" in content:
        new = re.sub(
            r"<!-- TWD_STATUS:BEGIN -->.*?<!-- TWD_STATUS:END -->",
            block,
            content,
            flags=re.S
        )
    else:
        # Append to end with a separator if file exists; else create
        sep = "\n\n---\n\n" if content.strip() else ""
        new = content + sep + block
    if new != content:
        file_path.write_text(new, encoding="utf-8")
        return True
    return False

def ensure_file(path: Path, title: str):
    if path.exists(): return
    path.write_text(f"# {title}\n\n", encoding="utf-8")

def main():
    block = stamp_block()
    ensure_file(GUIDE, "TrendWatchDesk — Operations Guide")
    ensure_file(README, "TrendWatchDesk — README")

    changed = False
    changed |= upsert_block(GUIDE, block)
    changed |= upsert_block(README, block)

    if changed:
        print("[info] docs updated")
        # stage from repo root
        os.chdir(ROOT)
        subprocess.run(["git", "add", "OPERATIONS_GUIDE.md", "README.md"], check=False)
    else:
        print("[info] docs already up-to-date; no changes")

if __name__ == "__main__":
    main()
