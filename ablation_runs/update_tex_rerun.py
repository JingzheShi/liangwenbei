"""Update one_page_summary.tex with rerun NN row 1-13 + rebuilt ensembles row 18-21.

Replaces the lines for the 13 NN rows and 4 ensemble rows with new aggregated numbers.
Computes Δ column as (this row mean) - (previous row mean). Row 1 Δ = '---'.
"""
import csv, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "onepage_v2" / "one_page_summary.tex"
NN_CSV = HERE / "rerun_13rows_results.csv"
ENS_DIR = HERE / "rerun_ensembles"

NN_LABELS = {
    1: "NN 154 raw + 分类 Loss (3-class CE)",
    2: "NN 154 raw + L2 regression Loss",
    3: "NN 154 raw + L2 + window-$z$ (输入归一化)",
    4: "+ LOB 派生 Feature",
    5: "+ 多尺度 OFI Feature",
    6: "+ 订单流强度 Feature",
    7: "+ 微结构波动 Feature",
    8: "+ 窗口统计 Feature",
    9: "+ 不对称性 Feature (370-d Feat)",
    10: "+ drop 11 feature w. KS 检验 (359-d Feat)",
    11: "+ 买卖镜像数据增强 (bid/ask flip, 强制方向对称)",
    12: "+ 稳健归一化 (sign-log1p on raw\\_last)",
    13: "+ train+val 全数据重训 (M7 retrain)",
}


def fmt_pnl(m, sd):
    if sd is None or sd == 0.0:
        return f"${m:+.2f}$"
    return f"${m:+.2f}{{\\scriptscriptstyle\\,\\pm {sd:.2f}}}$"


def fmt_delta(d):
    if d is None:
        return "---"
    sign = "+" if d >= 0 else "-"
    return f"${sign}{abs(d):.2f}$"


def main():
    # Load NN rerun results
    nn_means = {}
    nn_stds = {}
    with open(NN_CSV) as f:
        for row in csv.DictReader(f):
            r = int(row["row"])
            nn_means[r] = float(row["mean"]) if row["mean"] else None
            nn_stds[r] = float(row["std"]) if row["std"] else None

    # Load ensemble results
    def load_ens(name):
        p = ENS_DIR / name / "results.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())["decision"]["test_pnl_h60"]

    ens_rows = {
        18: ("(1$+$1) NN(row13) $+$ LGB(row17), sym EV", load_ens("row18_sym")),
        19: ("(5$+$5) NN(M7,同 HP) $+$ LGB(M7,同 HP), sym EV", load_ens("row19_sym")),
        20: ("(5$+$5) NN(M7,同 HP) $+$ LGB(M7,同 HP), asym EV", load_ens("row20_asym")),
        21: ("(50$+$50) NN(M7$+$SPO,HP cycling) $+$ LGB(M7,HP cycling), sym EV (SOTA)", load_ens("row21_sym")),
    }
    # Also fetch row 21 asym to report in caption
    row21_asym_pnl = load_ens("row21_asym")
    row21_sym_pnl = load_ens("row21_sym")

    # Read tex
    tex = TEX.read_text()

    # Build new NN row lines
    nn_lines = []
    for i in range(1, 14):
        m = nn_means[i]
        sd = nn_stds[i]
        if i == 1:
            delta = None
        else:
            prev = nn_means.get(i - 1)
            delta = (m - prev) if (m is not None and prev is not None) else None
        # Daggered label for row 13 (keep mark)
        label = NN_LABELS[i]
        if i == 13:
            label += "$^{\\dagger}$"
        pnl_str = fmt_pnl(m, sd) if m is not None else "TBD"
        delta_str = fmt_delta(delta)
        if i == 13:
            delta_str = delta_str + "$^{\\dagger}$"
        line = f"{i} & {label}                                                       & {pnl_str} & {delta_str} \\\\"
        nn_lines.append(line)

    # Build ensemble row lines (18-21) — Δ column references previous ensemble row
    ens_lines = []
    prev_ens_pnl = None
    for i in [18, 19, 20, 21]:
        label, m = ens_rows[i]
        if m is None:
            pnl_str = "TBD"
            delta_str = "---"
        else:
            pnl_str = f"${m:+.2f}$"
            if prev_ens_pnl is None:
                delta_str = "---"
            else:
                d = m - prev_ens_pnl
                sign = "+" if d >= 0 else "-"
                delta_str = f"${sign}{abs(d):.2f}$"
            prev_ens_pnl = m
        ens_lines.append(f"{i} & {label}                                                       & {pnl_str} & {delta_str} \\\\")

    # Find the table block in tex and replace
    # Pattern: between "% NN path block" and "\midrule" (LGB path)
    # Easier: identify by row number prefix at start of line (after possible whitespace) on rows 1-13 and 18-21.
    new_tex_lines = []
    in_nn = False
    in_ens = False
    nn_replaced = False
    ens_replaced = False
    for line in tex.splitlines():
        stripped = line.lstrip()
        # NN rows: detect "1 &", "2 &", ..., "13 &" patterns followed by content
        m_nn = re.match(r"^(\d{1,2})\s*&\s*(NN|\+).+\\\\\s*$", stripped)
        m_ens = re.match(r"^(\d{1,2})\s*&\s*\(.+\\\\\s*$", stripped)
        if m_nn:
            r = int(m_nn.group(1))
            if 1 <= r <= 13:
                if not nn_replaced:
                    for nl in nn_lines:
                        new_tex_lines.append(nl)
                    nn_replaced = True
                continue
        if m_ens:
            r = int(m_ens.group(1))
            if 18 <= r <= 21:
                if not ens_replaced:
                    for el in ens_lines:
                        new_tex_lines.append(el)
                    ens_replaced = True
                continue
        new_tex_lines.append(line)

    new_tex = "\n".join(new_tex_lines) + "\n"

    # Update caption: replace the dagger explanation block.
    # We'll regenerate a fresh caption that documents the unified rerun.
    sota_val = ens_rows[21][1]   # = row21_sym
    row13_mean = nn_means[13]
    row12_mean = nn_means[12]
    # Build new dagger sentence:
    if row12_mean is not None and row13_mean is not None:
        delta_m7 = row13_mean - row12_mean
        sign_m7 = "+" if delta_m7 >= 0 else "-"
        dagger_new = (
            f"$^{{\\dagger}}$Row 1--13 由\\textbf{{统一实现}}（unified script）重跑，固定 HP "
            f"(lr=$3{{\\times}}10^{{-4}}$, dropout 0.10, batch 4096, class-balanced sample weights, "
            f"AdamW $+$ CosineAnnealingLR, multiplicative noise scale $\\sim U(0.80,1.20)$), "
            f"每行 5 seeds (seed 1--5)。Row 13 (M7) 沿用 row 12 (V4) 的 best\\_ep $\\times 1.1$ 与 sym 阈值；"
            f"$\\Delta$ (M7$-$V4) $\\approx {sign_m7}{abs(delta_m7):.2f}$，与 LGB 同协议（row 16$\\to$17, $+4.30$）相比仍偏小——"
            f"M7 对 LGB 显著有效、对 NN 几乎无效，与项目历史一致。"
            f"Ensemble (rows 18--21) 公式 $\\hat p = (1.0\\overline{{\\hat p}}_{{\\mathrm{{NN}}}} + 1.5\\overline{{\\hat p}}_{{\\mathrm{{LGB}}}})/2.5$；"
            f"val pred 取 V4 模型 (M7 模型无独立 val)，test pred 取 M7 模型。"
            f"Row 18--20 取 NN row 13 五个 seed (同 HP=lr 3e-4/dr 0.10/bs 4096) 与 LGB M7 中 HP\\_CONFIGS[0] 的 5 个 seed (seed $\\in\\{{1,6,11,16,21\\}}$，对应同 HP)；"
            f"Row 21 用 50$\\times$50 异质集成 (HP cycling: NN 50 个 M7$+$SPO 模型 $\\times$ LGB 50 个 M7 模型)，"
            f"sym EV $=$ ${row21_sym_pnl:+.2f}$, asym EV $=$ ${row21_asym_pnl:+.2f}$ (val DE bounded); "
            f"表内 SOTA 采用 sym 以与 row 19 同阈值族对比。"
        )
    else:
        dagger_new = "$^{\\dagger}$Row 1--13 由统一实现重跑。"

    # Replace the caption block by finding the start of the existing caption (the
    # one immediately after the tabularx tabular for the big ablation) and replacing
    # the dagger explanation.
    pattern = re.compile(
        r"(\\captionof\{table\}\{\\textbf\{大 ablation：NN 渐进 \(1--13\)、LGB 渐进 \(14--17\)、Ensemble \(18--21\)\.\}\s+)"
        r"\$\^\{\\dagger\}\$.+?(结论：M7 对 LGB 显著有效，对 NN 几乎无效，与项目历史一致。Ensemble \(rows 18--21\) 全部使用 M7 retrained 基模型。\})",
        re.DOTALL,
    )
    new_caption_close = "Ensemble (rows 18--21) 全部使用 M7 retrained 基模型，详见上文。}"
    new_tex2, n = pattern.subn(lambda m: m.group(1) + dagger_new + "}", new_tex)
    if n == 0:
        # Caption may have been previously rewritten; try a looser match
        print("WARN: dagger replacement pattern did not match. Doing simple find/replace.")
        new_tex2 = new_tex.replace(
            "$^{\\dagger}$Row 13 的 V4 baseline",
            f"{dagger_new}\n% (legacy dagger note follows)\n$^{{\\dagger\\dagger}}$Row 13 的 V4 baseline"
        )

    TEX.write_text(new_tex2)
    print(f"Updated {TEX}")
    print(f"  NN rows replaced: {nn_replaced}")
    print(f"  Ensemble rows replaced: {ens_replaced}")
    print(f"  Caption dagger replaced: {n > 0}")
    print(f"\n  NEW NN row PnLs:")
    for i in range(1, 14):
        print(f"    row {i:2d}: mean={nn_means[i]:+.4f}  std={nn_stds[i]:.4f}")
    print(f"  NEW Ensemble row PnLs:")
    for i in [18, 19, 20, 21]:
        print(f"    row {i}: {ens_rows[i][1]:+.4f}")


if __name__ == "__main__":
    main()
