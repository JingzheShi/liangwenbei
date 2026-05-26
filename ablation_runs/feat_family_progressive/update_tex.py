"""Update one_page_summary.tex with:
- LGB path: add row '370-d SchemeP + LGB L2' between 226-d and 359-d (using 5-seed +27.28 ± 2.34)
- NN path: insert 6 new rows L2..L7 between current row 6 (NN 154+L2+window-z = +26.05) and row 7 (NN 359-d = +34.34)
- Renumber subsequent rows
- Change last row label from '(5+5) NN + LGB + SPO, asym EV' -> '(5+5) NN+SPO + LGB, asym EV'
- Update caption to reflect new structure
"""
import json
from pathlib import Path

WORKDIR = Path('/root/projects/liangwenbei_workdir')
TEX = WORKDIR / 'onepage_v2' / 'one_page_summary.tex'
AGG = WORKDIR / 'ablation_runs' / 'feat_family_progressive' / 'aggregated.json'

# 370-d LGB no-drop 5-seed
LGB_370 = (27.28, 2.34)

LABEL = {
    'L2': r'$+$ LOB 派生',
    'L3': r'$+$ 多尺度 OFI',
    'L4': r'$+$ 订单流强度',
    'L5': r'$+$ 微结构波动',
    'L6': r'$+$ 窗口统计',
    'L7': r'$+$ 不对称性 ($=$ 370-d, no drop)',
}

def fmt_mean_std(m, s):
    return f"$+{m:.2f}{{\\scriptscriptstyle\\,\\pm {s:.2f}}}$"

def fmt_delta(d):
    sign = '+' if d >= 0 else ''
    return f"${sign}{d:.2f}$"

def main():
    agg = json.load(open(AGG))
    text = TEX.read_text()
    nn_l1_mean = 26.05
    nn_l8 = (34.34, 0.50)
    lgb_226 = 24.11
    lgb_370 = LGB_370[0]
    lgb_359 = 26.58

    # =====================================================
    # Build new table body (rows 1..20)
    # =====================================================
    new_rows = []

    # ----- LGB path (4 rows) -----
    new_rows.append(r"\multicolumn{4}{@{}l}{\textit{\color{icmlblue}LGB path — 因子 $\to$ GBDT 渐进}} \\")
    new_rows.append(r"1  & 154 raw + LGB CE (3-class), high-conf gate                                    & $+21.95{\scriptscriptstyle\,\pm 1.16}$  & --- \\")
    new_rows.append(r"2  & 226-d SchemeC + LGB L2, sym EV                                                & $+24.11{\scriptscriptstyle\,\pm 0.26}$  & $+2.16$ \\")
    new_rows.append(rf"3  & 370-d SchemeP + LGB L2 (no drop)                                              & {fmt_mean_std(*LGB_370)}  & {fmt_delta(lgb_370 - lgb_226)} \\")
    new_rows.append(rf"4  & 359-d SchemeP + LGB L2                                                        & $+26.58{{\scriptscriptstyle\,\pm 0.96}}$  & {fmt_delta(lgb_359 - lgb_370)} \\")
    new_rows.append(r"\midrule")

    # ----- NN path (12 rows) -----
    new_rows.append(r"\multicolumn{4}{@{}l}{\textit{\color{icmlblue}NN path — 154 raw $\to$ 6 家族渐进 $\to$ 359 SchemeP，逐项加 aug（NN backbone）}$^{\dagger}$} \\")
    new_rows.append(r"5  & NN 154 raw + CE (3-cls), high-conf gate                                       & $+0.67{\scriptscriptstyle\,\pm 0.16}$   & (switch) \\")
    new_rows.append(r"6  & NN 154 raw + L2, sym EV                                                       & $+0.98{\scriptscriptstyle\,\pm 0.20}$   & $+0.31$ \\")
    new_rows.append(r"7  & NN 154 raw + L2 $+$ window-$z$                                                & $+26.05{\scriptscriptstyle\,\pm 0.57}$  & $\mathbf{+25.07}$ \\")
    prev = nn_l1_mean
    row_no = 8
    for L in ['L2', 'L3', 'L4', 'L5', 'L6', 'L7']:
        r = agg[L]
        m, s = r['mean'], r['std']
        d = m - prev
        new_rows.append(f"{row_no:<2} & {LABEL[L]:60s} & {fmt_mean_std(m,s)}  & {fmt_delta(d)} \\\\")
        prev = m
        row_no += 1
    # row 14 = NN 359-d (drop 11 from L7)
    new_rows.append(rf"14 & $+$ drop 11 ($=$ 359-d)                                                       & $+{nn_l8[0]:.2f}{{\scriptscriptstyle\,\pm {nn_l8[1]:.2f}}}$  & {fmt_delta(nn_l8[0] - prev)} \\")
    # row 15, 16 = aug rows (keep existing numbers)
    new_rows.append(r"15 & $+$ bid/ask mirror flip aug                                                   & $+34.10{\scriptscriptstyle\,\pm 1.90}$  & $-0.24$ \\")
    new_rows.append(r"16 & $+$ sign-log1p on rawlast                                                     & $+34.37{\scriptscriptstyle\,\pm 1.29}$  & $+0.27$ \\")
    new_rows.append(r"\midrule")

    # ----- Ensemble (4 rows) -----
    new_rows.append(r"\multicolumn{4}{@{}l}{\textit{\color{icmlblue}Ensemble — NN $+$ LGB 跨族}} \\")
    new_rows.append(r"17 & (1+1) NN $+$ LGB, sym EV                                                      & $+35.02$                                & $+0.65$ \\")
    new_rows.append(r"18 & (5+5) NN $+$ LGB, sym EV                                                      & $+36.07$                                & $+1.05$ \\")
    new_rows.append(r"19 & (5+5) NN $+$ LGB, asym EV                                                     & $+36.19$                                & $+0.12$ \\")
    new_rows.append(r"20 & (5+5) NN+SPO $+$ LGB, asym EV                                                 & $\mathbf{+37.73}$ (SOTA)                & $+1.54$ \\")

    new_table_body = "\n".join(new_rows)

    # ---- replace the table body (between \midrule after header and \bottomrule) ----
    import re
    # find the block to replace: from '\multicolumn{4}{@{}l}{\textit{\color{icmlblue}LGB path' through 'SPO, asym EV} ... \\\\'
    # Using regex with DOTALL.
    # The current block starts at '\multicolumn{4}{@{}l}{\textit{\color{icmlblue}LGB path' line (line 157)
    # and ends at row 13 (line 174) right before '\bottomrule' (line 175).
    pat = re.compile(
        r"(\\multicolumn\{4\}\{@\{\}l\}\{\\textit\{\\color\{icmlblue\}LGB path.*?asym EV\s*&\s*\$\\mathbf\{\+37\.73\}\$.*?\\\\)",
        re.DOTALL
    )
    if not pat.search(text):
        # try simpler boundary
        # Find start
        start_marker = r"\multicolumn{4}{@{}l}{\textit{\color{icmlblue}LGB path"
        end_marker_line = r"13 & (5+5) NN $+$ LGB $+$ SPO, asym EV"
        i = text.find(start_marker)
        j = text.find(r"\bottomrule", i)
        # j is the start of \bottomrule; back up to the prior \\
        # We'll just slice
        assert i >= 0 and j > i, "could not find table block"
        # Find last '\\\\\n' before j
        # Use last '\\\n' (line termination) before \bottomrule
        # Actually, simpler: replace everything from i to j-1
        new_text = text[:i] + new_table_body + "\n" + text[j:]
    else:
        new_text = pat.sub(new_table_body, text)

    # ---- update caption ----
    # Old caption:
    #   "\captionof{table}{\textbf{大 ablation（rows 1--9 = 5-seed mean ± std；rows 10--13 = NN + LGB 跨族 ensemble）.} $^{\dagger}$Rows 4--6 ..."
    new_caption = (
        r"\captionof{table}{\textbf{大 ablation（rows 1--16 = 5-seed mean $\pm$ std；rows 17--20 = NN $+$ LGB 跨族 ensemble）.} "
        r"$^{\dagger}$Rows 5--7 复刻 LGB 路径 154$\to$226 但用 NN backbone；rows 8--13 在 row 7 (NN 154+L2+window-$z$) 之上按六家族渐进添加（partition 互斥、并集 = 216 derived）；row 14 在 L7 (370-d no-drop) 之上 drop 11 KS-fail 列。"
        r"\textbf{Row 6$\to$7 加 window-$z$ 触发 $+25.07$ 跃升}（NN critical；LGB 同步骤 $\pm 0.4$ within noise）——首层 Linear 在 scale 失配输入上无法恢复，LayerNorm 在其后；树则 axis-aligned scale-invariant。"
        r"\textbf{Rows 17--20}：单 (1+1) 跨族 ensemble 即达 $+35.02$；扩到 $5\!+\!5$ multi-seed 再 $+1.05$ 至 $+36.07$，asym EV $+0.12$，SPO 决策焦点 $+1.54$，累计 $\mathbf{+37.73}$（SOTA）。LGB 项目 \texttt{HP\_CONFIGS[0]}。}"
    )
    # find old caption (single line, ends with closing brace)
    old_cap_pat = re.compile(r"\\captionof\{table\}\{\\textbf\{大 ablation.*?HP_CONFIGS.*?\}\}", re.DOTALL)
    if old_cap_pat.search(new_text):
        new_text = old_cap_pat.sub(new_caption, new_text)
    else:
        print("WARNING: could not find old caption to replace")

    TEX.write_text(new_text)
    print(f"wrote {TEX} ({len(new_text)} chars)")

if __name__ == '__main__':
    main()
