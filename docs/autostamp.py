#!/usr/bin/env python3
import os, datetime, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "OPERATIONS_GUIDE.md"
README = ROOT / "README.md"

def _env(k, default=""):
    v = os.getenv(k)
    return v if v is not None and v != "" else default

def stamp_block():
    now_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    mode         = _env("TWD_MODE")                 # charts | posters | all
    tf           = _env("TWD_TF", "D")              # D | W
    recency      = _env("TWD_BREAKING_RECENCY_MIN")
    min_sources  = _env("TWD_BREAKING_MIN_SOURCES")
    fallback     = _env("TWD_BREAKING_FALLBACK")
    allow_rss    = _env("TWD_ALLOW_RSS")
    wl           = _env("TWD_WATCHLIST")
    charts_br    = _env("TWD_CHARTS_BRANCH", "charts")
    posters_br   = _env("TWD_POSTERS_BRANCH", "posters")
    trigger_name = _env("TWD_TRIGGER_NAME")

    lines = []
    lines.append("<!-- TWD_STATUS:BEGIN -->")
    lines.append("")
    lines.append("## Automation Status (auto-generated)")
    lines.append(f"- **Last run:** {now_utc}")
    if trigger_name:
        lines.append(f"- **Triggered by:** {trigger_name}")
    if mode:
        lines.append(f"- **Mode:** `{mode}`   ·  **Timeframe:** `{tf}`")
    knobs = []
    if recency:     knobs.append(f"recency={recency}m")
    if min_sources: knobs.append(f"min_sources={min_sources}")
    if fallback:    knobs.append(f"fallback={fallback}")
    if allow_rss:   knobs.append(f"rss={allow_rss}")
    if knobs:
        lines.append(f"- **Breaking-posters knobs:** {', '.join(knobs)}")
    if wl:
        wl_list = [s.strip() for s in wl.split(",") if s.strip()]
        preview = ", ".join(wl_list[:20]) + (" …" if len(wl_list) > 20 else "")
        lines.append(f"- **Watchlist (preview):** {preview}")
    lines.append(f"- **Publish targets:** charts → `{charts_br}`, posters → `{posters_br}`")
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
        sep = "\n\n---\n\n" if content.strip() else ""
        new = content + sep + block
    if new != content:
        file_path.write_text(new, encoding="utf-8")
        return True
    return False

def ensure_file(path: Path, title: str):
    if not path.exists():
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
        os.chdir(ROOT)
        subprocess.run(["git", "add", "OPERATIONS_GUIDE.md", "README.md"], check=False)
    else:
        print("[info] docs already up-to-date; no changes")

if __name__ == "__main__":
    main()
