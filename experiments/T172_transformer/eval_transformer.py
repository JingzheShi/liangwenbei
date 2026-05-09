"""T172b: Eval transformer ensemble on holdout and build v2N_Tr zip.

Architecture: GroupTransformer (d_model=64, nhead=4, 2 layers, n_tokens=32)

Eval is IN-SAMPLE for transformer (trained date 0-119, test date 96-119).
Uses PyTorch CUDA for fast inference in eval. Predictor.py uses torch CPU.
"""
from __future__ import annotations
import json, os, hashlib, shutil, zipfile
import numpy as np
import pandas as pd
import torch

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE   = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
V2_PKG  = os.path.join(ROOT, "experiments", "R_full_retrain", "pkg_iter019_v2")
OUT_DIR = HERE
SEEDS   = (1, 7, 13, 42, 100)
FEE     = 0.0001

DROP_NAMES = [
    "dualz_ask_diff1","dualz_bid_diff5","dualz_ask_diff5",
    "qrank_W100_spread1","qrank_W100_spread5","qrank_W100_spread10","qrank_W100_cumspread",
    "kyle_lam_W50","kyle_lam_W100","roll_eff_spr_ratio_W100","liq_asym_top5_W5",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)


# ──────────────────────────── PyTorch transformer ─────────────────────────────

class GroupTransformerTorch(torch.nn.Module):
    """GroupTransformer forward pass in PyTorch (loads from npz)."""

    def __init__(self, npz_path: str):
        super().__init__()
        d = np.load(npz_path, allow_pickle=True)
        self.n_layers = int(d["n_layers"][0])
        self.n_heads  = int(d["n_heads"][0])
        self.n_tokens = int(d["n_tokens"][0])
        self.d_token  = int(d["d_token"][0])
        self.d_head   = self.d_token // self.n_heads
        self.clip     = float(d["clip"][0])
        self.target_scale = float(d["target_scale"][0])
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
        import torch.nn.functional as F
        Xs = X_full[:, self.keep_idx]
        Xs = (Xs - self.feat_mean) / self.feat_std
        Xs = torch.nan_to_num(Xs, nan=0.0)
        Xs = Xs.clamp(-self.clip, self.clip)

        B = Xs.shape[0]
        D, H, DH, S = self.d_token, self.n_heads, self.d_head, self.n_tokens + 1

        tokens = F.linear(Xs, self.proj_w, self.proj_b).reshape(B, self.n_tokens, D)
        tokens = F.layer_norm(tokens, (D,), self.proj_nw, self.proj_nb)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, S, D)

        for i in range(self.n_layers):
            attn_in_w  = getattr(self, f"L{i}_attn_in_proj_w")
            attn_in_b  = getattr(self, f"L{i}_attn_in_proj_b")
            attn_out_w = getattr(self, f"L{i}_attn_out_w")
            attn_out_b = getattr(self, f"L{i}_attn_out_b")
            ffn1_w = getattr(self, f"L{i}_ffn1_w")
            ffn1_b = getattr(self, f"L{i}_ffn1_b")
            ffn2_w = getattr(self, f"L{i}_ffn2_w")
            ffn2_b = getattr(self, f"L{i}_ffn2_b")
            n1w = getattr(self, f"L{i}_norm1_w"); n1b = getattr(self, f"L{i}_norm1_b")
            n2w = getattr(self, f"L{i}_norm2_w"); n2b = getattr(self, f"L{i}_norm2_b")
            FF = ffn1_w.shape[0]

            # Pre-LN attention
            h = F.layer_norm(tokens, (D,), n1w, n1b)
            qkv = F.linear(h.reshape(B*S, D), attn_in_w, attn_in_b).reshape(B, S, 3*D)
            Q = qkv[:,:,:D].reshape(B,S,H,DH).permute(0,2,1,3).reshape(B*H, S, DH)
            K = qkv[:,:,D:2*D].reshape(B,S,H,DH).permute(0,2,1,3).reshape(B*H, S, DH)
            V = qkv[:,:,2*D:].reshape(B,S,H,DH).permute(0,2,1,3).reshape(B*H, S, DH)
            scores = torch.bmm(Q, K.transpose(1,2)) / (DH ** 0.5)
            attn_w = torch.softmax(scores, dim=-1)
            out = torch.bmm(attn_w, V).reshape(B, H, S, DH).permute(0,2,1,3).reshape(B*S, D)
            attn_out = F.linear(out, attn_out_w, attn_out_b).reshape(B, S, D)
            tokens = tokens + attn_out

            # Pre-LN FFN
            h = F.layer_norm(tokens, (D,), n2w, n2b)
            h = F.gelu(F.linear(h.reshape(B*S, D), ffn1_w, ffn1_b)).reshape(B, S, FF)
            h = F.linear(h.reshape(B*S, FF), ffn2_w, ffn2_b).reshape(B, S, D)
            tokens = tokens + h

        cls_out = tokens[:, 0, :]
        h = F.gelu(F.linear(cls_out, self.head_fc1_w, self.head_fc1_b))
        h = F.layer_norm(h, (D,), self.head_ln_w, self.head_ln_b)
        return (F.linear(h, self.head_fc2_w, self.head_fc2_b).squeeze(-1) / self.target_scale)


def get_pred_tr_torch(X_full_np: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Predict transformer ensemble (5 seeds) on X_full using CUDA."""
    X_gpu = torch.from_numpy(X_full_np.astype(np.float32)).to(DEVICE)
    N = X_full_np.shape[0]
    pred_acc = np.zeros(N, dtype=np.float32)
    for s in SEEDS:
        model = GroupTransformerTorch(os.path.join(HERE, f"nn_tr_h60_seed{s}.npz")).to(DEVICE)
        model.eval()
        preds = []
        for start in range(0, N, batch_size):
            xb = X_gpu[start:start+batch_size]
            with torch.no_grad():
                preds.append(model(xb).cpu().numpy())
        pred_s = np.concatenate(preds)
        pred_acc += pred_s
        print(f"    seed={s}  mean={pred_s.mean():+.6f}", flush=True)
        del model
        torch.cuda.empty_cache()
    return (pred_acc / len(SEEDS)).astype(np.float32)


# ──────────────────────────── numpy helpers (for v2 MLP) ──────────────────────

def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))

def _layernorm_np(x, w, b, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True); v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b

class _MLPNumpy:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=False)
        self.in_dim = int(d["in_dim"][0]); self.hidden = list(d["hidden"].tolist())
        self.use_layernorm = bool(d["use_layernorm"][0])
        self.target_scale = float(d["target_scale"][0])
        self.feat_mean = d["feat_mean"].astype(np.float32); self.feat_std = d["feat_std"].astype(np.float32)
        self.keep_idx = d["keep_idx"].astype(np.int64); self.clip = float(d["clip"][0])
        self.W, self.b, self.LN_W, self.LN_b = [], [], [], []
        for i in range(len(self.hidden)):
            self.W.append(d[f"L{i}_W"].astype(np.float32)); self.b.append(d[f"L{i}_b"].astype(np.float32))
            if self.use_layernorm:
                self.LN_W.append(d[f"LN{i}_W"].astype(np.float32)); self.LN_b.append(d[f"LN{i}_b"].astype(np.float32))
        self.WF = d["LF_W"].astype(np.float32); self.bF = d["LF_b"].astype(np.float32)

    def predict(self, X):
        Xs = X if X.shape[1] == self.in_dim else X[:, self.keep_idx]
        Xs = np.clip(np.nan_to_num((Xs.astype(np.float32) - self.feat_mean) / self.feat_std), -self.clip, self.clip)
        h = Xs
        for i in range(len(self.hidden)):
            h = h @ self.W[i].T + self.b[i]
            if self.use_layernorm: h = _layernorm_np(h, self.LN_W[i], self.LN_b[i])
            h = _gelu_tanh(h)
        return (h @ self.WF.T + self.bF).squeeze(-1).astype(np.float32) / self.target_scale


# ──────────────────────────── data / eval helpers ─────────────────────────────

def load_test_data():
    d = np.load(os.path.join(CACHE, "schemeP_test.npz"))
    with open(os.path.join(CACHE, "schemeP_feat_names.txt")) as f:
        all_names = [l.strip() for l in f]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(len(all_names)) if i not in drop_idx], dtype=np.int64)
    X_full = d['X']                                  # (N, 370) raw
    X      = d['X'][:, keep_idx].astype(np.float32) # (N, 359)
    sym    = d['sym'].astype(np.int64)
    mp_t   = d['mp_t'].astype(np.float64)
    mp_th  = d['mp_t60'].astype(np.float64)
    return X_full, X, sym, mp_t, mp_th


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    fee  = FEE * np.abs(side) * np.abs(mp_th + 1.0 + mp_t + 1.0)
    return (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)


def eval_pnl(pred, sym, mp_t, mp_th, thr_up, thr_dn,
             per_sym_beta=None, per_sym_sigma=None, default_beta=0.16, default_sigma=3.998e-4):
    SYMS = sorted(np.unique(sym).tolist())
    pnls = []
    for s in SYMS:
        mask = sym == s; p = pred[mask]
        if per_sym_beta is not None:
            beta = per_sym_beta.get(s, default_beta)
            sigma = per_sym_sigma.get(s, default_sigma) if per_sym_sigma else default_sigma
            band = beta * sigma
        else:
            band = 0.0
        action = np.full(mask.sum(), 1, dtype=np.int8)
        action[p >  (thr_up + band)] = 2
        action[p < -(thr_dn + band)] = 0
        pnls.append(float(vectorized_pnl(action, mp_t[mask], mp_th[mask]).sum()))
    return sum(pnls), pnls


# ──────────────────────────── main ─────────────────────────────────────────────

def main():
    THR_UP = 0.00029995433796130955
    THR_DN = 0.0002158528296175958
    PER_SYM_BETA  = {0: 0.10, 1: 0.50, 2: 0.30, 3: 0.00, 4: 0.00}
    PER_SYM_SIGMA = {0: 0.0002403901, 1: 0.0004712397, 2: 0.0004522216,
                     3: 0.0004235249, 4: 0.0004279811}

    print("Loading test data ...", flush=True)
    X_full, X, sym, mp_t, mp_th = load_test_data()
    print(f"  N={len(X):,}  X_full={X_full.shape}", flush=True)

    print("\nPred LGB (5 seeds) ...", flush=True)
    import lightgbm as lgb
    pred_lgb = None
    for s in SEEDS:
        b = lgb.Booster(model_file=os.path.join(V2_PKG, f"model_h60_seed{s}.txt"))
        p = b.predict(X).astype(np.float32)
        pred_lgb = p if pred_lgb is None else pred_lgb + p
        print(f"  seed={s}  mean={p.mean():+.6f}", flush=True)
    pred_lgb /= len(SEEDS)

    print("\nPred NN T87 (5 seeds numpy) ...", flush=True)
    pred_nn = None
    for s in SEEDS:
        p = _MLPNumpy(os.path.join(V2_PKG, f"nn_h60_seed{s}.npz")).predict(X)
        pred_nn = p if pred_nn is None else pred_nn + p
        print(f"  seed={s}  mean={p.mean():+.6f}", flush=True)
    pred_nn /= len(SEEDS)

    print(f"\nPred Transformer T172 (5 seeds, torch on {DEVICE}) ...", flush=True)
    import time
    t0 = time.time()
    pred_tr = get_pred_tr_torch(X_full)
    print(f"  Transformer pred done in {time.time()-t0:.1f}s  mean={pred_tr.mean():+.6f}  std={pred_tr.std():.6f}", flush=True)

    # Baseline v2 (w_nn=1.0, w_lgb=0.5)
    pred_v2 = (pred_nn + 0.5 * pred_lgb) / 1.5
    pnl_v2, per_v2 = eval_pnl(pred_v2, sym, mp_t, mp_th, THR_UP, THR_DN,
                                PER_SYM_BETA, PER_SYM_SIGMA)
    print(f"\n[v2 IN-SAMPLE] total={pnl_v2:+.4f}  per_sym={[f'{v:+.4f}' for v in per_v2]}", flush=True)

    pnl_tr, per_tr = eval_pnl(pred_tr, sym, mp_t, mp_th, THR_UP, THR_DN,
                               PER_SYM_BETA, PER_SYM_SIGMA)
    print(f"[Tr-only IN-SAMPLE] total={pnl_tr:+.4f}  per_sym={[f'{v:+.4f}' for v in per_tr]}", flush=True)

    # Ensemble sweep (w_nn, w_tr, w_lgb)
    configs = {
        "1.0:1.0:0.5": (1.0, 1.0, 0.5),
        "1.0:0.5:0.5": (1.0, 0.5, 0.5),
        "1.5:1.0:0.5": (1.5, 1.0, 0.5),
        "1.0:1.0:1.0": (1.0, 1.0, 1.0),
        "1.0:0.5:1.0": (1.0, 0.5, 1.0),
        "1.5:1.0:1.0": (1.5, 1.0, 1.0),
        "2.0:1.0:0.5": (2.0, 1.0, 0.5),
        "1.0:2.0:0.5": (1.0, 2.0, 0.5),
    }
    print("\n=== Ensemble sweep ===", flush=True)
    best_pnl, best_cfg, best_w = pnl_v2, "v2_baseline", (1.0, 0.0, 0.5)
    ens_results = {}
    for name, (w_nn, w_tr, w_lgb) in configs.items():
        ws = w_nn + w_tr + w_lgb
        p = (w_nn*pred_nn + w_tr*pred_tr + w_lgb*pred_lgb) / ws
        pnl, per = eval_pnl(p, sym, mp_t, mp_th, THR_UP, THR_DN, PER_SYM_BETA, PER_SYM_SIGMA)
        delta = pnl - pnl_v2
        print(f"  {name:<16} {pnl:+.4f}  delta={delta:+.4f}  per={[f'{v:+.4f}' for v in per]}", flush=True)
        ens_results[name] = {"pnl": pnl, "per_sym": per, "weights": [w_nn, w_tr, w_lgb]}
        if pnl > best_pnl:
            best_pnl, best_cfg, best_w = pnl, name, (w_nn, w_tr, w_lgb)

    print(f"\nBest: {best_cfg}  pnl={best_pnl:+.4f}  delta={best_pnl-pnl_v2:+.4f}", flush=True)

    results = {
        "task": "T172b finalize transformer ensemble",
        "remote_models_synced": True,
        "v2_baseline_pnl_insample": pnl_v2,
        "transformer_only_pnl_insample": pnl_tr,
        "ensemble_configs": {k: {"pnl": v["pnl"], "weights": v["weights"]} for k, v in ens_results.items()},
        "best_config": best_cfg,
        "best_pnl_insample": best_pnl,
        "delta_vs_v2": best_pnl - pnl_v2,
        "best_weights": {"w_nn": best_w[0], "w_tr": best_w[1], "w_lgb": best_w[2]},
        "warning": "Transformer trained 0-119; holdout in-sample. Platform eval required for true validation.",
    }

    # Build zip if delta > 0.5
    build = best_pnl > pnl_v2 + 0.5
    print(f"\nBuild zip: {build}  (need delta > 0.5, got {best_pnl-pnl_v2:+.4f})", flush=True)
    if build:
        w_nn, w_tr, w_lgb = best_w
        zp = build_zip(w_nn, w_tr, w_lgb, PER_SYM_BETA, PER_SYM_SIGMA, THR_UP, THR_DN)
        md5 = hashlib.md5(open(zp,"rb").read()).hexdigest()
        results["zip"] = zp; results["md5"] = md5
        print(f"  zip={zp}  md5={md5}", flush=True)
        os.makedirs("/tmp/metabot-outputs/worker-4799145f", exist_ok=True)
        shutil.copy2(zp, "/tmp/metabot-outputs/worker-4799145f/")
    else:
        results["zip"] = None

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results.json", flush=True)
    print(f"\nRESULT: task=T172b metrics={{v2_pnl={pnl_v2:.4f},tr_pnl={pnl_tr:.4f},best_ens={best_pnl:.4f},delta={best_pnl-pnl_v2:+.4f}}} notes=[best={best_cfg},in-sample-insample]", flush=True)


def build_zip(w_nn, w_tr, w_lgb, per_sym_beta, per_sym_sigma, thr_up, thr_dn):
    pkg_dir = os.path.join(HERE, "pkg_v2N_Tr")
    os.makedirs(pkg_dir, exist_ok=True)
    for fn in os.listdir(V2_PKG):
        src = os.path.join(V2_PKG, fn)
        if os.path.isfile(src) and not fn.endswith(".pyc"):
            shutil.copy2(src, pkg_dir)
    for s in SEEDS:
        shutil.copy2(os.path.join(HERE, f"nn_tr_h60_seed{s}.npz"), pkg_dir)

    thresholds = {
        "_doc": f"v2N_Tr: NN(w={w_nn})+Transformer(w={w_tr})+LGB(w={w_lgb}). Transformer trained 0-119 (in-sample).",
        "horizons": [
            {"h": 5,  "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 10, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 20, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 40, "thr_up": 1.0, "thr_dn": 1.0, "active": False},
            {"h": 60, "thr_up": thr_up, "thr_dn": thr_dn,
             "w_nn": w_nn, "w_lgb": w_lgb, "w_tr": w_tr,
             "active": True, "ensemble_seeds": list(SEEDS)},
        ],
        "conformal_wrapper": {
            "enabled": True,
            "per_sym_beta": {str(k): v for k, v in per_sym_beta.items()},
            "default_beta_for_ood": 0.16,
            "per_sym_sigma": {str(k): v for k, v in per_sym_sigma.items()},
            "default_sigma_for_ood": 3.998317e-4,
        }
    }
    with open(os.path.join(pkg_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    _write_predictor(pkg_dir, w_nn, w_tr, w_lgb)

    import subprocess
    r = subprocess.run(["python3", os.path.join(pkg_dir, "Predictor.py")], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Smoke test failed:\n{r.stderr}\n{r.stdout}")
    print(f"  Smoke test: {r.stdout.strip()}", flush=True)

    zip_name = "submission_050909_iter019_v2N_transformer.zip"
    zip_path = os.path.join(HERE, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(pkg_dir)):
            fp = os.path.join(pkg_dir, fn)
            if os.path.isfile(fp) and not fn.endswith(".pyc"):
                zf.write(fp, fn)
    print(f"  Built {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)", flush=True)
    return zip_path


def _write_predictor(pkg_dir, w_nn, w_tr, w_lgb):
    """Copy static Predictor_v2N_Tr.py to pkg_dir as Predictor.py."""
    src = os.path.join(HERE, "Predictor_v2N_Tr.py")
    shutil.copy2(src, os.path.join(pkg_dir, "Predictor.py"))




if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()

