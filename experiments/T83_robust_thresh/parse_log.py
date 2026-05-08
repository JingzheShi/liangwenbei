"""Parse run_main.log into results_main.json (used by pick_winner.py)."""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(path):
    with open(path) as f:
        lines = f.readlines()
    records = []
    full_repl = None
    cur_split = None
    for ln in lines:
        m = re.search(r"iter_013 thr replication: ([0-9.]+)", ln)
        if m:
            full_repl = float(m.group(1))
        m = re.search(r"=== Split: (\S+) \|", ln)
        if m:
            cur_split = m.group(1)
            continue
        # match "  [name] thr=(a,b) train=X eval=Y gap=G"
        m = re.match(r"\s+\[([\w\d_.]+)\]\s+thr=\(([\d.e\-+]+),([\d.e\-+]+)\)\s+train=([\-\d.]+)\s+eval=([\-\d.]+)\s+gap=([\-\d.]+)", ln)
        if m:
            name = m.group(1)
            thr_up = float(m.group(2))
            thr_dn = float(m.group(3))
            train_pnl = float(m.group(4))
            eval_pnl = float(m.group(5))
            gap = float(m.group(6))
            records.append({
                "split": cur_split, "strategy": name,
                "thr_up": thr_up, "thr_dn": thr_dn,
                "train_pnl": train_pnl, "eval_pnl": eval_pnl,
                "gap": gap,
            })
    return full_repl, records


if __name__ == "__main__":
    full_repl, records = parse(os.path.join(HERE, "run_main.log"))
    out = {"iter013_full_replication": full_repl, "records": records}
    with open(os.path.join(HERE, "results_main.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Parsed {len(records)} records, full_repl={full_repl}")
    print("Splits:", set(r["split"] for r in records))
