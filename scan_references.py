#!/usr/bin/env python3
"""
scan_references.py — Extract all references from workdir markdown/tex/py files.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

WORKDIR = Path("/root/projects/liangwenbei_workdir")

SKIP_DIRS = {
    "_data_unpacked", ".git", "node_modules", "_remote_logs_archive",
    "worktrees", "__pycache__", "factor_per_feature_audit",
    "ablation_runs", "agent_timeline",
}

SCAN_EXTS = {".md", ".tex"}

PATTERNS = {
    "arxiv_id": re.compile(r"arXiv[:\s]+(\d{4}\.\d{4,5})", re.IGNORECASE),
    "arxiv_url": re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE),
    "latex_cite": re.compile(r"\\cite(?:\[.*?\])?\{([^}]+)\}"),
    "kaggle_url": re.compile(r"https?://www\.kaggle\.com/[^\s\)\"'<>]+"),
    "github_url": re.compile(r"https?://github\.com/[^\s\)\"'<>]+"),
    "arxiv_full_url": re.compile(r"https?://arxiv\.org/[^\s\)\"'<>]+"),
    "ssrn_url": re.compile(r"https?://ssrn\.com/[^\s\)\"'<>]+"),
    "medium_url": re.compile(r"https?://[^\s]*medium\.com/[^\s\)\"'<>]+"),
    "tds_url": re.compile(r"https?://towardsdatascience\.com/[^\s\)\"'<>]+"),
}

def extract_context(lines, idx, window=2):
    start = max(0, idx - window)
    end = min(len(lines), idx + window + 1)
    return " | ".join(l.strip() for l in lines[start:end] if l.strip())

def scan_file(filepath):
    results = []
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        lines = content.split("\n")
    except Exception:
        return results

    rel_path = str(filepath.relative_to(WORKDIR))
    for line_idx, line in enumerate(lines):
        for pat_name, pat in PATTERNS.items():
            for m in pat.finditer(line):
                ctx = extract_context(lines, line_idx)
                results.append({
                    "type": pat_name,
                    "match": m.group(0),
                    "group1": m.group(1) if m.lastindex and m.lastindex >= 1 else None,
                    "file": rel_path,
                    "line": line_idx + 1,
                    "context": ctx[:400],
                })
    return results

def walk_files(base):
    for root, dirs, files in os.walk(base):
        # Prune skip dirs in place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            p = Path(root) / fname
            if p.suffix.lower() in SCAN_EXTS:
                yield p

def main():
    all_results = []
    files_scanned = 0

    for filepath in sorted(walk_files(WORKDIR)):
        results = scan_file(filepath)
        if results:
            all_results.extend(results)
            files_scanned += 1
        else:
            files_scanned += 1

    print(f"Scanned {files_scanned} files, found {len(all_results)} raw matches")

    # Group by (type, key), collect all files
    grouped = defaultdict(lambda: {"files": [], "contexts": []})
    for r in all_results:
        key = (r["type"], r.get("group1") or r["match"])
        if r["file"] not in grouped[key]["files"]:
            grouped[key]["files"].append(r["file"])
        grouped[key]["contexts"].append(r["context"][:200])
        grouped[key]["type"] = r["type"]
        grouped[key]["match"] = r.get("group1") or r["match"]

    output = []
    for key, data in sorted(grouped.items()):
        ctx_deduped = list(dict.fromkeys(data["contexts"]))[:3]
        output.append({
            "type": data["type"],
            "key": data["match"],
            "files": data["files"][:5],
            "contexts": ctx_deduped,
        })

    with open(WORKDIR / "references_raw.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(output)} unique references to references_raw.json")

    type_counts = defaultdict(int)
    for item in output:
        type_counts[item["type"]] += 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

if __name__ == "__main__":
    main()
