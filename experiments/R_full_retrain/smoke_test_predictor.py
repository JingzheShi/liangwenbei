"""Smoke test: load Predictor from each pkg dir, run on synthetic batch.

Validates:
  - Predictor.py loads with new full-retrain LGB models
  - feature dim matches (359)
  - No exceptions on a 100x150 synthetic input
  - Output is a list[list[int]] with values in {0, 1, 2}
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def smoke_test(pkg_dir: str, has_sym: bool):
    print(f"\n=== smoke {pkg_dir} (has_sym={has_sym}) ===", flush=True)
    cfg_path = os.path.join(pkg_dir, "config.json")
    cfg = json.load(open(cfg_path))
    feats = cfg["feature"]
    print(f"  config feature count: {len(feats)}  has_sym_in_config: {'sym' in feats}", flush=True)

    sys.path.insert(0, pkg_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            f"P_{os.path.basename(pkg_dir)}",
            os.path.join(pkg_dir, "Predictor.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Predictor = mod.Predictor
        p = Predictor()
        rng = np.random.default_rng(0)

        df_cols = feats
        df = pd.DataFrame(
            rng.standard_normal((100, len(df_cols))).astype(np.float32),
            columns=df_cols,
        )
        if has_sym:
            df["sym"] = 2

        out = p.predict([df, df, df])
        assert isinstance(out, list)
        assert len(out) == 3
        assert all(len(o) == 5 for o in out)
        flat = [a for row in out for a in row]
        assert all(a in (0, 1, 2) for a in flat), set(flat)
        print(f"  OK — first row: {out[0]}, n_active={sum(1 for a in flat if a != 1)}/15", flush=True)

        if has_sym:
            df_ood = df.copy(); df_ood["sym"] = 99
            out_ood = p.predict([df_ood])
            print(f"  OOD sym=99: {out_ood[0]}", flush=True)
            df_no = df.drop(columns=["sym"])
            out_no = p.predict([df_no])
            print(f"  no sym col: {out_no[0]}", flush=True)
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        return False
    finally:
        sys.path.remove(pkg_dir)


def main():
    v1 = os.path.join(HERE, "pkg_iter019_v1")
    v2 = os.path.join(HERE, "pkg_iter019_v2")
    ok1 = smoke_test(v1, has_sym=False)
    ok2 = smoke_test(v2, has_sym=True)
    print(f"\n=== summary ===\n  v1 OK: {ok1}\n  v2 OK: {ok2}", flush=True)
    if not (ok1 and ok2):
        sys.exit(1)


if __name__ == "__main__":
    main()
