"""Partition 216 derived features into 6 mutually-exclusive families and
build cumulative L1..L8 keep-index sets in 370-d schemeP space.

Rationale for partition (judgement calls documented in feat_family_audit.md):
  - LOB 派生   : wmp_*                                    (= micro_price family)
  - 多尺度 OFI : mlofi_*, ewma_ofi_*, kyle_lam_*, ofi_tox_*
  - 订单流强度  : ewma_a*_*_intst (24), cancel_imb_*       (= intst + cancel pressure)
  - 微结构波动  : rv_*, signed_rv_*, kyle_inv_*, rskew_*, vol_burst_*, amt_burst_*,
                rv_ratio_*, jshare_*, roll_eff_spr_*, signed_bv_*, spread_reg_*
  - 窗口统计   : dualz_*, qrank_*, adapt_mom_*, trade_pers_*
  - 不对称性   : gofi_*, mid_ewma_resid_*, liq_asym_*

L8 drops 11 KS-fail features (T59_FAIL_NAMES + STAGE5_FAIL_NAMES).
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
SCHEME_FEAT_NAMES = HERE.parent / "cache_log1p" / "schemeP_feat_names.txt"
EXTRA_FEAT_NAMES  = HERE.parent / "cache_log1p" / "schemeP_extra_feat_names.txt"

KS_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5",
]


def classify(name: str) -> str:
    # LOB 派生
    if name.startswith("wmp_"):
        return "LOB"
    # 多尺度 OFI
    if name.startswith("mlofi_") or name.startswith("ewma_ofi_") \
            or name.startswith("kyle_lam_") or name.startswith("ofi_tox_"):
        return "OFI"
    # 订单流强度
    if name.startswith("ewma_a") and "_intst" in name:
        return "INTST"
    if name.startswith("cancel_imb_"):
        return "INTST"
    # 微结构波动
    for p in ("rv_w", "signed_rv_", "kyle_inv_", "rskew_",
              "vol_burst_", "amt_burst_", "rv_ratio_",
              "jshare_", "roll_eff_spr_", "signed_bv_", "spread_reg_"):
        if name.startswith(p):
            return "VOL"
    # 窗口统计
    if name.startswith("dualz_") or name.startswith("qrank_") \
            or name.startswith("adapt_mom_") or name.startswith("trade_pers_"):
        return "WIN"
    # 不对称性
    if name.startswith("gofi_") or name.startswith("mid_ewma_resid_") \
            or name.startswith("liq_asym_"):
        return "ASYM"
    return "UNCLASSIFIED"


def main() -> None:
    with open(SCHEME_FEAT_NAMES) as f:
        all_names_370 = [l.strip() for l in f if l.strip()]
    with open(EXTRA_FEAT_NAMES) as f:
        derived_216 = [l.strip() for l in f if l.strip()]

    assert len(all_names_370) == 370, f"expected 370, got {len(all_names_370)}"
    assert len(derived_216) == 216, f"expected 216, got {len(derived_216)}"
    # schemeP cache: first 154 raw, then 216 derived
    assert all_names_370[154:] == derived_216, "schemeP_feat_names tail != derived 216"

    raw154_idx = list(range(154))

    families = {"LOB": [], "OFI": [], "INTST": [], "VOL": [], "WIN": [], "ASYM": []}
    unclassified = []
    for i, name in enumerate(derived_216):
        fam = classify(name)
        global_idx = 154 + i  # in 370-d space
        if fam == "UNCLASSIFIED":
            unclassified.append((i, name))
        else:
            families[fam].append((global_idx, name))

    assert not unclassified, f"unclassified features: {unclassified}"

    # validate counts sum to 216
    total = sum(len(v) for v in families.values())
    assert total == 216, f"family sum {total} != 216"

    # validate mutual exclusion (already enforced by single-pass classify)
    seen = set()
    for fam, items in families.items():
        for gi, name in items:
            assert gi not in seen, f"{name} (idx {gi}) duplicate"
            seen.add(gi)

    # cumulative levels
    order = ["LOB", "OFI", "INTST", "VOL", "WIN", "ASYM"]
    levels = {}
    levels["L1"] = sorted(raw154_idx)
    cum_global = list(raw154_idx)
    for k, fam in enumerate(order, start=2):
        cum_global = cum_global + [gi for gi, _ in families[fam]]
        levels[f"L{k}"] = sorted(cum_global)
    # L7 == 154+216 = 370
    assert len(levels["L7"]) == 370, f"L7={len(levels['L7'])}"
    # L8 = L7 minus 11 KS-fail
    name_to_idx = {n: i for i, n in enumerate(all_names_370)}
    ks_idx = {name_to_idx[n] for n in KS_FAIL_NAMES if n in name_to_idx}
    assert len(ks_idx) == 11, f"expected 11 KS-fail indices, got {len(ks_idx)}"
    levels["L8"] = sorted(i for i in levels["L7"] if i not in ks_idx)
    assert len(levels["L8"]) == 359

    out = {
        "all_names_370": all_names_370,
        "derived_216": derived_216,
        "ks_fail_idx_in_370": sorted(ks_idx),
        "ks_fail_names": [all_names_370[i] for i in sorted(ks_idx)],
        "families": {fam: [(int(gi), n) for gi, n in items] for fam, items in families.items()},
        "family_counts": {fam: len(items) for fam, items in families.items()},
        "level_dims": {lv: len(idx) for lv, idx in levels.items()},
        "levels": {lv: [int(i) for i in idx] for lv, idx in levels.items()},
        "family_order": order,
    }
    with open(HERE / "partition.json", "w") as f:
        json.dump(out, f, indent=2)
    print("=== Family counts ===")
    for fam in order:
        print(f"  {fam:6s} : {len(families[fam]):3d}")
    print(f"  TOTAL  : {total}")
    print("=== Level dims ===")
    for lv in ["L1","L2","L3","L4","L5","L6","L7","L8"]:
        print(f"  {lv} : {len(levels[lv])}")
    print(f"\nSaved -> {HERE/'partition.json'}")

    # also save human-readable feature lists per family
    with open(HERE / "feature_families.md", "w") as f:
        f.write("# 6-Family Partition of 216 Derived Features\n\n")
        for fam in order:
            f.write(f"## {fam} ({len(families[fam])})\n")
            for gi, name in families[fam]:
                f.write(f"- `{name}` (idx {gi})\n")
            f.write("\n")
        f.write(f"## KS-fail (dropped in L8) ({len(ks_idx)})\n")
        for n in sorted(out["ks_fail_names"]):
            f.write(f"- `{n}`\n")


if __name__ == "__main__":
    main()
