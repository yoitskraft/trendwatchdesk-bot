#!/usr/bin/env python3
# scripts/update_docs.py
import os, re, subprocess, sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
OPS    = os.path.join(ROOT, "OPERATIONS_GUIDE.md")
MAIN   = os.path.join(ROOT, "main.py")
LOGOS  = os.path.join(ROOT, "assets", "logos")

def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, cwd=ROOT, text=True).strip()

def repo_slug() -> str:
    return os.environ.get("GITHUB_REPOSITORY", "your/repo")

def branch_list():
    try:
        out = sh("git branch -r | sed 's#origin/##' | sort -u")
        return [b for b in out.splitlines() if b]
    except Exception:
        return []

def repo_tree():
    try:
        out = sh(r"""git ls-files | sed 's|^\./||' | awk -F/ '{
            d=""; for(i=1;i<NF;i++){d=(d?d"/":"")$i; a[d]=1}
        } END{
            print "."; for (k in a) print k
        }' | sort""")
        return out
    except Exception:
        return "."

def read(p): 
    return open(p, "r", encoding="utf-8").read() if os.path.exists(p) else ""

def write_if_changed(p, new_text) -> bool:
    old = read(p)
    if old == new_text:
        return False
    with open(p, "w", encoding="utf-8") as f:
        f.write(new_text)
    return True

def patch_block(full_text: str, block: str, new_body: str) -> str:
    start = f"<!-- AUTOGEN:{block}:START -->"
    end   = f"<!-- AUTOGEN:{block}:END -->"
    if start not in full_text or end not in full_text:
        add = f"\n\n{start}\n{new_body.rstrip()}\n{end}\n"
        return (full_text.rstrip() + add) if full_text else (start + "\n" + new_body.rstrip() + "\n" + end + "\n")
    import re as _re
    pat = _re.compile(re.escape(start) + r".*?" + re.escape(end), _re.S)
    return pat.sub(start + "\n" + new_body.rstrip() + "\n" + end, full_text, count=1)

def parse_pools_watchlist():
    pools = {}
    watch = set()
    txt = read(MAIN)
    if not txt:
        return pools, sorted(watch)
    m = re.search(r"POOLS\s*=\s*\{(.*?)\}\s*", txt, flags=re.S)
    if m:
        body = m.group(1)
        for k, arr in re.findall(r"['\"]([^'\"]+)['\"]\s*:\s*\[([^\]]*)\]", body):
            syms = re.findall(r"['\"]([A-Za-z0-9\.\-]+)['\"]", arr)
            pools[k] = sorted(list(dict.fromkeys(syms)))
            watch.update(syms)
    mw = re.search(r"WATCHLIST\s*=\s*\[([^\]]*)\]", txt)
    if mw:
        syms = re.findall(r"['\"]([A-Za-z0-9\.\-]+)['\"]", mw.group(1))
        watch.update(syms)
    return pools, sorted(watch)

def logo_coverage(tickers):
    have = set()
    if os.path.isdir(LOGOS):
        for f in os.listdir(LOGOS):
            if f.lower().endswith(".png"):
                have.add(os.path.splitext(f)[0].upper())
    missing = [t for t in tickers if t.upper() not in have]
    return sorted(have), missing

def main():
    repo = repo_slug()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    branches = branch_list()
    tree     = repo_tree()
    pools, watch = parse_pools_watchlist()
    have, missing = logo_coverage(watch)

    # Build blocks
    badges = " ".join([
        f"![Daily](https://github.com/{repo}/actions/workflows/daily.yml/badge.svg)",
        f"![Docs](https://github.com/{repo}/actions/workflows/docs-autostamp.yml/badge.svg)"
    ])
    branches_md = "\n".join(f"- `{b}`" for b in branches) or "_(no remote branches found)_"
    pools_md = "\n".join(f"- **{k}**: {', '.join(v)}" for k,v in sorted(pools.items())) or "_(no pools found)_"
    logos_md = (
        f"**Have ({len(have)})**: {', '.join(have) if have else '—'}\n\n"
        f"**Missing ({len(missing)})**: {', '.join(missing) if missing else '—'}"
    )
    tree_md = f"```\n{tree}\n```"
    stamp = f"_Last updated: **{now}**_"

    # README
    readme = read(README) or "# TrendWatchDesk\n"
    readme = patch_block(readme, "BADGES", badges)
    readme = patch_block(readme, "BRANCHES", branches_md)
    readme = patch_block(readme, "POOLS", pools_md)
    readme = patch_block(readme, "LOGOS", logos_md)
    readme = patch_block(readme, "TREE", tree_md)
    readme = patch_block(readme, "STAMP", stamp)

    # OPS
    ops = read(OPS) or "# TrendWatchDesk Operations Guide\n"
    ops = patch_block(ops, "STRUCTURE", tree_md)
    ops = patch_block(ops, "BRANCHES", branches_md)
    ops = patch_block(ops, "POOLS", pools_md)
    ops = patch_block(ops, "LOGOS", logos_md)
    ops = patch_block(ops, "STAMP", f"_Doc auto-stamped: **{now}**_")

    ch_readme = write_if_changed(README, readme)
    ch_ops    = write_if_changed(OPS, ops)
    print(f"[docs] README updated: {ch_readme}, OPERATIONS_GUIDE updated: {ch_ops}")
    sys.exit(0)

if __name__ == "__main__":
    main()
