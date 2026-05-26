"""Aggregate 5-seed test PnL per row from rerun_13rows. Emit CSV."""
import csv, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "rerun_13rows"
OUT_CSV = HERE / "rerun_13rows_results.csv"

ROW_LABEL = {
    1: "NN 154 raw + 3-class CE (high-conf gate)",
    2: "NN 154 raw + L2 (no window-z, no mirror, no log1p)",
    3: "NN 154 raw + L2 + window-z",
    4: "NN 165-d L2 (+ LOB 派生)",
    5: "NN 213-d L3 (+ 多尺度 OFI)",
    6: "NN 240-d L4 (+ 订单流强度)",
    7: "NN 272-d L5 (+ 微结构波动)",
    8: "NN 335-d L6 (+ 窗口统计)",
    9: "NN 370-d L7 (+ 不对称性, 全 370)",
    10: "NN 359-d L8 (drop 11 KS-fail)",
    11: "NN 359-d L8 + mirror flip aug",
    12: "NN 359-d L8 + mirror + sign-log1p on raw_last",
    13: "NN 359-d L8 + mirror + log1p + M7 retrain",
}


def main():
    rows_out = []
    for row in range(1, 14):
        vals = []
        for s in [1, 2, 3, 4, 5]:
            p = SRC / f"row{row:02d}_s{s}" / "results.json"
            if p.exists():
                r = json.loads(p.read_text())
                vals.append(r["test_pnl_reported"])
            else:
                vals.append(None)
        valid = [v for v in vals if v is not None]
        n = len(valid)
        if n == 0:
            row_data = {"row": row, "label": ROW_LABEL[row],
                        "s1": None, "s2": None, "s3": None, "s4": None, "s5": None,
                        "n": 0, "mean": None, "std": None}
        else:
            mean = sum(valid) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in valid) / max(1, n - 1))
            row_data = {"row": row, "label": ROW_LABEL[row],
                        "s1": vals[0], "s2": vals[1], "s3": vals[2],
                        "s4": vals[3], "s5": vals[4],
                        "n": n, "mean": round(mean, 4), "std": round(std, 4)}
        rows_out.append(row_data)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row", "label",
                                          "s1", "s2", "s3", "s4", "s5",
                                          "n", "mean", "std"])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"Wrote {OUT_CSV}")
    print(f"\n  Row | n | mean (test PnL)  | std    | label")
    print(f"  ----|---|------------------|--------|------")
    for r in rows_out:
        seeds_str = " ".join(f"{v:.2f}" if v is not None else "----" for v in [r["s1"], r["s2"], r["s3"], r["s4"], r["s5"]])
        if r["mean"] is None:
            print(f"  {r['row']:2d}  | {r['n']} | --             | --     | {r['label']}")
        else:
            print(f"  {r['row']:2d}  | {r['n']} | {r['mean']:+8.4f}        | {r['std']:.4f} | {r['label']}")
        print(f"         seeds=[{seeds_str}]")
    return rows_out


if __name__ == "__main__":
    main()
