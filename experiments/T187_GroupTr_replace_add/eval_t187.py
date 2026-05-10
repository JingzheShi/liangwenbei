"""T187: Holdout eval (dates 96-119) for Plan A (replace) and Plan B (add) packages.

Uses schemeP_test.npz (N=442080, 370 raw features).
Models: T172b GroupTr 5-seed + v2 LGB + v2 NNs.
v2 thresholds: w_lgb=1.5, thr_up=0.000300, thr_dn=0.000216, beta_1=0.40.
"""
from __future__ import annotations
import json, os, hashlib, shutil, zipfile, time
import numpy as np
import torch
import torch.nn.functional as F
import lightgbm as lgb

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE   = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
V2_PKG  = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")
T172    = os.path.join(ROOT, "experiments", "T172_transformer")

SEEDS   = (1, 7, 13, 42, 100)
FEE     = 0.0001
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DROP_NAMES = [
    "dualz_ask_diff1","dualz_bid_diff5","dualz_ask_diff5",
    "qrank_W100_spread1","qrank_W100_spread5","qrank_W100_spread10","qrank_W100_cumspread",
    "kyle_lam_W50","kyle_lam_W100","roll_eff_spr_ratio_W100","liq_asym_top5_W5",
]

THR_UP = 0.00029995433796130955
THR_DN = 0.0002158528296175958
PER_SYM_BETA  = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.00, 4: 0.00}
PER_SYM_SIGMA = {0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
                 3: 0.0004235249, 4: 0.0004279811}
DEFAULT_BETA  = 0.16
DEFAULT_SIGMA = 0.0003998317

print(f"Using device: {DEVICE}")


# ──────────────────── torch GroupTransformer inference ────────────────────────

class GroupTransformerTorch(torch.nn.Module):
    def __init__(self, npz_path: str):
        super().__init__()
        d = np.load(npz_path, allow_pickle=True)
        self.n_layers = int(d["n_layers"][0]); self.n_heads = int(d["n_heads"][0])
        self.n_tokens = int(d["n_tokens"][0]); self.d_token = int(d["d_token"][0])
        self.d_head = self.d_token // self.n_heads
        self.clip = float(d["clip"][0]); self.target_scale = float(d["target_scale"][0])
        self.register_buffer("feat_mean", torch.from_numpy(d["feat_mean"].astype(np.float32)))
        self.register_buffer("feat_std",  torch.from_numpy(d["feat_std"].astype(np.float32)))
        self.register_buffer("keep_idx",  torch.from_numpy(d["keep_idx"].astype(np.int64)))
        self.register_buffer("proj_w",    torch.from_numpy(d["proj_w"].astype(np.float32)))
        self.register_buffer("proj_b",    torch.from_numpy(d["proj_b"].astype(np.float32)))
        self.register_buffer("proj_nw",   torch.from_numpy(d["proj_norm_w"].astype(np.float32)))
        self.register_buffer("proj_nb",   torch.from_numpy(d["proj_norm_b"].astype(np.float32)))
        self.register_buffer("cls_token", torch.from_numpy(d["cls_token"].astype(np.float32)))
        for i in range(self.n_layers):
            for k in ["attn_in_proj_w","attn_in_proj_b","attn_out_w","attn_out_b",
                      "ffn1_w","ffn1_b","ffn2_w","ffn2_b","norm1_w","norm1_b","norm2_w","norm2_b"]:
                self.register_buffer(f"L{i}_{k}", torch.from_numpy(d[f"L{i}_{k}"].astype(np.float32)))
        self.register_buffer("head_fc1_w", torch.from_numpy(d["head_fc1_w"].astype(np.float32)))
        self.register_buffer("head_fc1_b", torch.from_numpy(d["head_fc1_b"].astype(np.float32)))
        self.register_buffer("head_ln_w",  torch.from_numpy(d["head_ln_w"].astype(np.float32)))
        self.register_buffer("head_ln_b",  torch.from_numpy(d["head_ln_b"].astype(np.float32)))
        self.register_buffer("head_fc2_w", torch.from_numpy(d["head_fc2_w"].astype(np.float32)))
        self.register_buffer("head_fc2_b", torch.from_numpy(d["head_fc2_b"].astype(np.float32)))

    @torch.no_grad()
    def forward(self, X_full: torch.Tensor) -> torch.Tensor:
        Xs = X_full[:, self.keep_idx]
        Xs = ((Xs - self.feat_mean) / self.feat_std).nan_to_num_(0.0).clamp_(-self.clip, self.clip)
        B, D, H, DH = Xs.shape[0], self.d_token, self.n_heads, self.d_head
        S = self.n_tokens + 1
        tok = F.layer_norm(F.linear(Xs, self.proj_w, self.proj_b).reshape(B, self.n_tokens, D),
                           (D,), self.proj_nw, self.proj_nb)
        tok = torch.cat([self.cls_token.expand(B, -1, -1), tok], dim=1)
        for i in range(self.n_layers):
            aiw = getattr(self, f"L{i}_attn_in_proj_w"); aib = getattr(self, f"L{i}_attn_in_proj_b")
            aow = getattr(self, f"L{i}_attn_out_w");     aob = getattr(self, f"L{i}_attn_out_b")
            f1w = getattr(self, f"L{i}_ffn1_w"); f1b = getattr(self, f"L{i}_ffn1_b")
            f2w = getattr(self, f"L{i}_ffn2_w"); f2b = getattr(self, f"L{i}_ffn2_b")
            n1w = getattr(self, f"L{i}_norm1_w"); n1b = getattr(self, f"L{i}_norm1_b")
            n2w = getattr(self, f"L{i}_norm2_w"); n2b = getattr(self, f"L{i}_norm2_b")
            FF = f1w.shape[0]
            h = F.layer_norm(tok, (D,), n1w, n1b)
            qkv = F.linear(h.reshape(B*S, D), aiw, aib).reshape(B, S, 3*D)
            Q = qkv[:,:,:D].reshape(B,S,H,DH).permute(0,2,1,3).reshape(B*H,S,DH)
            K = qkv[:,:,D:2*D].reshape(B,S,H,DH).permute(0,2,1,3).reshape(B*H,S,DH)
            V = qkv[:,:,2*D:].reshape(B,S,H,DH).permute(0,2,1,3).reshape(B*H,S,DH)
            attn_w = torch.softmax(torch.bmm(Q, K.transpose(1,2)) / (DH**0.5), dim=-1)
            attn_out = F.linear(torch.bmm(attn_w, V).reshape(B,H,S,DH).permute(0,2,1,3).reshape(B*S,D),
                                aow, aob).reshape(B,S,D)
            tok = tok + attn_out
            h = F.layer_norm(tok, (D,), n2w, n2b)
            tok = tok + F.linear(F.gelu(F.linear(h.reshape(B*S,D),f1w,f1b)).reshape(B,S,FF),
                                 f2w,f2b).reshape(B,S,D)
        h = F.gelu(F.linear(tok[:,0,:], self.head_fc1_w, self.head_fc1_b))
        h = F.layer_norm(h, (D,), self.head_ln_w, self.head_ln_b)
        return (F.linear(h, self.head_fc2_w, self.head_fc2_b).squeeze(-1) / self.target_scale)


def get_pred_tr(X_full_np: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    X_t = torch.from_numpy(X_full_np.astype(np.float32)).to(DEVICE)
    N = X_full_np.shape[0]
    acc = np.zeros(N, dtype=np.float32)
    for s in SEEDS:
        model = GroupTransformerTorch(os.path.join(T172, f"nn_tr_h60_seed{s}.npz")).to(DEVICE).eval()
        preds = []
        for st in range(0, N, batch_size):
            preds.append(model(X_t[st:st+batch_size]).cpu().numpy())
        ps = np.concatenate(preds)
        acc += ps
        print(f"    Tr seed={s}  mean={ps.mean():+.6f}", flush=True)
        del model
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return (acc / len(SEEDS)).astype(np.float32)


# ──────────────────── numpy MLP inference ─────────────────────────────────────

def _gelu(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))

def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b

class _MLPNumpy:
    def __init__(self, p):
        d = np.load(p, allow_pickle=False)
        self.hidden = list(d["hidden"].tolist()); self.uln = bool(d["use_layernorm"][0])
        self.ts = float(d["target_scale"][0]); self.fm = d["feat_mean"].astype(np.float32)
        self.fs = d["feat_std"].astype(np.float32); self.ki = d["keep_idx"].astype(np.int64)
        self.clip = float(d["clip"][0])
        self.W, self.b, self.LW, self.Lb = [], [], [], []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{i}_W"].astype(np.float32)); self.b.append(d[f"L{i}_b"].astype(np.float32))
            if self.uln: self.LW.append(d[f"LN{i}_W"].astype(np.float32)); self.Lb.append(d[f"LN{i}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32); self.bF = d["LF_b"].astype(np.float32)
    def predict(self, X):
        Xf = X if X.shape[1] == len(self.ki) else X[:, self.ki]
        Xs = np.clip(np.nan_to_num((Xf.astype(np.float32) - self.fm) / self.fs), -self.clip, self.clip)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.uln: h = _ln(h, self.LW[i], self.Lb[i])
            h = _gelu(h)
        return ((h @ self.WF.T + self.bF).squeeze(-1) / self.ts).astype(np.float32)

def get_pred_nn(X_kept: np.ndarray) -> np.ndarray:
    acc = None
    for s in SEEDS:
        m = _MLPNumpy(os.path.join(V2_PKG, f"nn_h60_seed{s}.npz"))
        p = m.predict(X_kept)
        acc = p if acc is None else acc + p
        print(f"    NN seed={s}  mean={p.mean():+.6f}", flush=True)
    return (acc / len(SEEDS)).astype(np.float32)

def get_pred_lgb(X_kept: np.ndarray) -> np.ndarray:
    acc = None
    for s in SEEDS:
        b = lgb.Booster(model_file=os.path.join(V2_PKG, f"model_h60_seed{s}.txt"))
        p = b.predict(X_kept).astype(np.float32)
        acc = p if acc is None else acc + p
        print(f"    LGB seed={s}  mean={p.mean():+.6f}", flush=True)
    return (acc / len(SEEDS)).astype(np.float32)


# ──────────────────── data loading ────────────────────────────────────────────

def load_test():
    d = np.load(os.path.join(CACHE, "schemeP_test.npz"))
    with open(os.path.join(CACHE, "schemeP_feat_names.txt")) as f:
        all_names = [l.strip() for l in f]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    drop_set = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(len(all_names)) if i not in drop_set], dtype=np.int64)
    X_full = d["X"].astype(np.float32)
    X_kept = X_full[:, keep_idx]
    sym    = d["sym"].astype(np.int64)
    mp_t   = d["mp_t"].astype(np.float64)
    mp_th  = d["mp_t60"].astype(np.float64)
    return X_full, X_kept, sym, mp_t, mp_th


# ──────────────────── PnL eval ────────────────────────────────────────────────

def pnl_total(pred: np.ndarray, sym: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray,
              w_nn: float = 0.0, w_tr: float = 0.0, w_lgb: float = 0.0,
              label: str = "") -> float:
    total = 0.0
    per_sym = []
    for s in sorted(np.unique(sym)):
        m = sym == s; p = pred[m]
        beta = PER_SYM_BETA.get(int(s), DEFAULT_BETA)
        sigma = PER_SYM_SIGMA.get(int(s), DEFAULT_SIGMA)
        band = beta * sigma
        act = np.full(m.sum(), 1, dtype=np.int8)
        act[p > (THR_UP + band)] = 2
        act[p < -(THR_DN + band)] = 0
        side = act.astype(np.float64) - 1.0
        fee = FEE * np.abs(side) * np.abs(mp_th[m] + 1.0 + mp_t[m] + 1.0)
        sym_pnl = float((side * (mp_th[m] - mp_t[m]) - fee).sum() / (mp_t[m] + 1.0).mean())
        total += sym_pnl
        per_sym.append(sym_pnl)
    print(f"  [{label}] total={total:+.4f}  per_sym={[f'{v:+.4f}' for v in per_sym]}")
    return total


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_zip(pkg_dir: str, zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(pkg_dir)):
            if fn.startswith("__"): continue
            fp = os.path.join(pkg_dir, fn)
            if os.path.isfile(fp):
                zf.write(fp, fn)
    m = md5_file(zip_path)
    print(f"  Built {os.path.basename(zip_path)}  md5={m}  size={os.path.getsize(zip_path)//1024}KB")
    return m


# ──────────────────── main ────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading test data (dates 96-119) ...", flush=True)
    X_full, X_kept, sym, mp_t, mp_th = load_test()
    print(f"  N={len(X_full):,}  X_full={X_full.shape}  X_kept={X_kept.shape}", flush=True)

    print("\nPredicting LGB (5 seeds) ...", flush=True)
    pred_lgb = get_pred_lgb(X_kept)

    print("\nPredicting NN T87 v2 (5 seeds) ...", flush=True)
    pred_nn = get_pred_nn(X_kept)

    print(f"\nPredicting GroupTransformer T172b (5 seeds, {DEVICE}) ...", flush=True)
    pred_tr = get_pred_tr(X_full)
    print(f"  Tr done in {time.time()-t0:.1f}s", flush=True)

    # Reference: v2 baseline (w_nn=1.0, w_lgb=1.5, w_tr=0) with correct betas
    pred_v2 = (1.0 * pred_nn + 1.5 * pred_lgb) / 2.5
    pnl_v2 = pnl_total(pred_v2, sym, mp_t, mp_th, label="v2-baseline w_nn=1.0 w_lgb=1.5")

    # Plan A: Replace T87 with GroupTr (w_tr=1.0, w_lgb=1.5)
    pred_A = (1.0 * pred_tr + 1.5 * pred_lgb) / 2.5
    pnl_A = pnl_total(pred_A, sym, mp_t, mp_th, label="Plan-A replace w_tr=1.0 w_lgb=1.5")

    # Plan B NN-heavy: w_nn=1.5, w_tr=1.0, w_lgb=1.0
    pred_B_heavy = (1.5 * pred_nn + 1.0 * pred_tr + 1.0 * pred_lgb) / 3.5
    pnl_B_heavy = pnl_total(pred_B_heavy, sym, mp_t, mp_th, label="Plan-B-nn-heavy 1.5:1.0:1.0")

    # Plan B equal: w_nn=1.0, w_tr=1.0, w_lgb=1.0
    pred_B_equal = (1.0 * pred_nn + 1.0 * pred_tr + 1.0 * pred_lgb) / 3.0
    pnl_B_equal = pnl_total(pred_B_equal, sym, mp_t, mp_th, label="Plan-B-equal 1.0:1.0:1.0")

    # Also compute transformer-only (as standalone reference)
    pnl_tr_only = pnl_total(pred_tr, sym, mp_t, mp_th, label="GroupTr-only w_tr=1.0")

    print(f"\n=== SUMMARY ===")
    T170_INSAMPLE = 149.95  # known SOTA in-sample
    print(f"  v2 baseline: {pnl_v2:+.4f}  (Δ vs T170={pnl_v2 - T170_INSAMPLE:+.4f})")
    print(f"  GroupTr-only: {pnl_tr_only:+.4f}  (Δ vs T170={pnl_tr_only - T170_INSAMPLE:+.4f})")
    print(f"  Plan A (replace): {pnl_A:+.4f}  (Δ vs T170={pnl_A - T170_INSAMPLE:+.4f})")
    print(f"  Plan B nn-heavy: {pnl_B_heavy:+.4f}  (Δ vs T170={pnl_B_heavy - T170_INSAMPLE:+.4f})")
    print(f"  Plan B equal: {pnl_B_equal:+.4f}  (Δ vs T170={pnl_B_equal - T170_INSAMPLE:+.4f})")

    # Build zips
    WDIR = HERE
    OUTDIR = os.path.abspath(os.path.join(ROOT, "..")) if False else ROOT  # store in workdir root
    # Store zips in workdir root (same as other submissions)
    zip_A = os.path.join(ROOT, "submission_050910_iter019_v2N_GroupTr_replace.zip")
    zip_B_heavy = os.path.join(ROOT, "submission_050910_iter019_v2N_GroupTr_add.zip")
    zip_B_equal = os.path.join(ROOT, "submission_050910_iter019_v2N_GroupTr_add_equal.zip")

    print("\nBuilding zips ...")
    md5_A = build_zip(os.path.join(WDIR, "pkg_A_replace"), zip_A)
    md5_B_heavy = build_zip(os.path.join(WDIR, "pkg_B_nn_heavy"), zip_B_heavy)
    md5_B_equal = build_zip(os.path.join(WDIR, "pkg_B_equal"), zip_B_equal)

    results = {
        "task": "T187 GroupTr replace + add to v2 stack",
        "models_source": "T172b existing (5-seed GroupTransformer trained date 0-119)",
        "eval_dates": "96-119 (in-sample: both NNs and LGB trained on 0-119)",
        "warning": "In-sample eval. Platform = true validation.",
        "T170_insample_reference": T170_INSAMPLE,
        "v2_baseline": {
            "holdout_pnl": round(pnl_v2, 4),
            "weights": "w_nn=1.0 w_lgb=1.5 w_tr=0",
            "note": "v2 NNs + v2 LGB + v2 betas (beta_1=0.40)"
        },
        "grouptr_standalone": {
            "holdout_pnl": round(pnl_tr_only, 4),
            "delta_vs_T170": round(pnl_tr_only - T170_INSAMPLE, 4)
        },
        "plan_A_replace": {
            "holdout_pnl": round(pnl_A, 4),
            "delta_vs_T170": round(pnl_A - T170_INSAMPLE, 4),
            "delta_vs_v2": round(pnl_A - pnl_v2, 4),
            "weights": "w_tr=1.0 w_lgb=1.5 (no MLP)",
            "zip": "submission_050910_iter019_v2N_GroupTr_replace.zip",
            "md5": md5_A
        },
        "plan_B_add_nn_heavy": {
            "holdout_pnl": round(pnl_B_heavy, 4),
            "delta_vs_T170": round(pnl_B_heavy - T170_INSAMPLE, 4),
            "delta_vs_v2": round(pnl_B_heavy - pnl_v2, 4),
            "weights": "w_nn=1.5 w_tr=1.0 w_lgb=1.0",
            "zip": "submission_050910_iter019_v2N_GroupTr_add.zip",
            "md5": md5_B_heavy
        },
        "plan_B_add_equal": {
            "holdout_pnl": round(pnl_B_equal, 4),
            "delta_vs_T170": round(pnl_B_equal - T170_INSAMPLE, 4),
            "delta_vs_v2": round(pnl_B_equal - pnl_v2, 4),
            "weights": "w_nn=1.0 w_tr=1.0 w_lgb=1.0",
            "zip": "submission_050910_iter019_v2N_GroupTr_add_equal.zip",
            "md5": md5_B_equal
        }
    }

    out_path = os.path.join(WDIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    main()
