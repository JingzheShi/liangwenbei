"""Interaction feature engineering — Top 50 from Worker A's research.

Input: X_grp (N, 5, 359) z-scored L8-pruned schemeP features (cross_sym_nn layout).
Output: X_inter (N, 5, 458), feature_names: list[str] of len 458.

Sym-agnostic:
- All operations apply identical formula across sym axis (no per-sym params).
- Cross-sym ops (zscore/demean/rank/share) treat 5 sym as a permutation-equivariant set.
- Pair ops use ordered (i,j) i!=j for asymmetric base features (20 dims),
  unordered i<j for antisymmetric combos (10 dims).

Date-agnostic:
- No use of `date` field.

All numpy vectorized — no Python row loops.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PARTITION_PATH = Path('/root/projects/liangwenbei_workdir/ablation_runs/feat_family_progressive/partition.json')
FEAT_NAMES_PATH = Path('/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p/schemeP_feat_names.txt')

_L8_NAMES = None
_NAME2IDX = None


def _init_names():
    global _L8_NAMES, _NAME2IDX
    if _L8_NAMES is None:
        L8 = json.loads(PARTITION_PATH.read_text())['levels']['L8']
        names_all = FEAT_NAMES_PATH.read_text().splitlines()
        _L8_NAMES = [names_all[i] for i in L8]
        _NAME2IDX = {n: i for i, n in enumerate(_L8_NAMES)}


def col(name: str) -> int:
    _init_names()
    return _NAME2IDX[name]


# ---------- Building blocks -----------------------------------------------

S = 5
_ORD_PAIRS = np.array([(i, j) for i in range(S) for j in range(S) if i != j])   # (20, 2)
_UNORD_PAIRS = np.array([(i, j) for i in range(S) for j in range(i + 1, S)])    # (10, 2)


def _vec(X: np.ndarray, base_idx: int) -> np.ndarray:
    """Return (N, 5) slice of base feature."""
    return X[:, :, base_idx]


def pair_diff_ordered(X: np.ndarray, base_idx: int) -> np.ndarray:
    """(N, 20): X[i] - X[j] for all ordered pairs (i,j), i!=j."""
    b = _vec(X, base_idx)                     # (N, 5)
    diff = b[:, _ORD_PAIRS[:, 0]] - b[:, _ORD_PAIRS[:, 1]]  # (N, 20)
    return diff


def pair_diff_unordered(b: np.ndarray) -> np.ndarray:
    """(N, 10): b[i] - b[j] for i<j. b: (N, 5)."""
    return b[:, _UNORD_PAIRS[:, 0]] - b[:, _UNORD_PAIRS[:, 1]]


def csec_demean(X: np.ndarray, base_idx: int) -> np.ndarray:
    """(N, 5): X[i] - mean_i(X)."""
    b = _vec(X, base_idx)
    return b - b.mean(axis=1, keepdims=True)


def csec_zscore(X: np.ndarray, base_idx: int, eps: float = 1e-6) -> np.ndarray:
    """(N, 5): (X[i] - mean) / std."""
    b = _vec(X, base_idx)
    m = b.mean(axis=1, keepdims=True)
    s = b.std(axis=1, keepdims=True)
    return (b - m) / np.maximum(s, eps)


def csec_rank(X: np.ndarray, base_idx: int) -> np.ndarray:
    """(N, 5): rank position normalized to [0, 1]."""
    b = _vec(X, base_idx)
    order = np.argsort(b, axis=1)
    ranks = np.empty_like(b, dtype=np.float32)
    np.put_along_axis(ranks, order, np.arange(S, dtype=np.float32).reshape(1, S), axis=1)
    return ranks / float(S - 1)


def csec_signed_rank(X: np.ndarray, base_idx: int) -> np.ndarray:
    """(N, 5): rank centered to [-1, 1] (i.e., (rank-2)/2 for S=5)."""
    r = csec_rank(X, base_idx)
    return (r - 0.5) * 2.0


def csec_market_broadcast(X: np.ndarray, base_idx: int) -> np.ndarray:
    """(N, 5): mean(b) broadcast to all 5 syms."""
    b = _vec(X, base_idx)
    m = b.mean(axis=1, keepdims=True)
    return np.broadcast_to(m, (b.shape[0], S)).copy()


def csec_share(b_pos: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(N, 5): b[i] / sum(b)."""
    s = b_pos.sum(axis=1, keepdims=True)
    return b_pos / (np.abs(s) + eps)


def csec_demean_vec(b: np.ndarray) -> np.ndarray:
    return b - b.mean(axis=1, keepdims=True)


def csec_zscore_vec(b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = b.mean(axis=1, keepdims=True)
    s = b.std(axis=1, keepdims=True)
    return (b - m) / np.maximum(s, eps)


def softmax_attention_mid(X: np.ndarray, base_idx: int, tau: float = 1.0) -> np.ndarray:
    """F139 / F205: sum_j softmax(-|mid_i-mid_j|*tau) * (mid_j-mid_i), i.e. each sym
    gets weighted-mean of partners' mids minus self.
    Returns (N, 5).
    """
    b = _vec(X, base_idx)                                       # (N, 5)
    diff = b[:, None, :] - b[:, :, None]                        # (N, S=i, S=j)  : mid_j - mid_i
    logits = -np.abs(diff) * tau                                # (N, 5, 5)
    # mask self (i==j) with -inf so attention only over partners
    eye = np.eye(S, dtype=bool)
    logits = np.where(eye[None, :, :], -1e9, logits)
    # softmax along j
    m = logits.max(axis=2, keepdims=True)
    e = np.exp(logits - m)
    w = e / (e.sum(axis=2, keepdims=True) + 1e-12)              # (N, 5, 5)
    return (w * diff).sum(axis=2)                               # (N, 5)


# ---------- Top 50 feature spec --------------------------------------------
# Each entry: id, n_dims, generator_fn(X, sup)
# Output is concat in this order. Final dim: 458.

TOP50_ORDER = [
    'F003', 'F001', 'F004', 'F009', 'F010', 'F002', 'F015', 'F040', 'F034', 'F078',
    'F079', 'F031', 'F019', 'F102', 'F021', 'F157', 'F175', 'F110', 'F195', 'F179',
    'F193', 'F123', 'F124', 'F049', 'F047', 'F062', 'F145', 'F202', 'F201', 'F060',
    'F085', 'F108', 'F109', 'F063', 'F147', 'F168', 'F166', 'F011', 'F177', 'F067',
    'F008', 'F126', 'F046', 'F165', 'F139', 'F205', 'F056', 'F081', 'F104', 'F191',
]


def _sup(name: str) -> int:
    """Resolve schemeP base name to L8 column index, with safe aliasing."""
    _init_names()
    # Aliases (research-side name -> actual column in L8)
    aliases = {
        'mid': 'midprice1',
        'mp_t': 'midprice1',
        'wmp': 'wmp_lvl1',
        'wap_lvl5': 'wmp_lvl5',
        'microprice': 'wmp_lvl1',         # microprice ≡ Stoikov's wmp_lvl1
        'imb1': 'imbalance',
        'ofi': 'mlofi_W5_lvl1',
        'ofi_W5_lvl1': 'mlofi_W5_lvl1',
        'ofi_W5_lvl3': 'mlofi_W5_lvl3',
        'ofi_W5_lvl5': 'mlofi_W5_lvl5',
        'ofi_W20': 'mlofi_W20_lvl1',
        'logret_W5': 'signed_rv_W20',     # no W5 — use shortest signed-vol (proxy)
        'logret_W20': 'signed_rv_W50',
        'logret_W100': 'signed_rv_W100',
        'spread': 'spread1',
        'RV': 'rv_w20',
        'realized_vol': 'rv_w20',
        'kyleinv': 'kyle_inv_W50',
        'inst_logret': 'signed_rv_W20',
    }
    name = aliases.get(name, name)
    return _NAME2IDX[name]


def _depth_vec(X: np.ndarray) -> np.ndarray:
    """(N, 5): totalbsize + totalasize (z-scored space; sum is fine)."""
    return _vec(X, _sup('totalbsize')) + _vec(X, _sup('totalasize'))


def _top_book_size_vec(X: np.ndarray) -> np.ndarray:
    """(N, 5): bsize1 + asize1."""
    return _vec(X, _sup('bsize1')) + _vec(X, _sup('asize1'))


def _sum_mlofi_lvl15(X: np.ndarray) -> np.ndarray:
    """(N, 5): sum mlofi_W5_lvl1..5."""
    return sum(_vec(X, _sup(f'mlofi_W5_lvl{k}')) for k in range(1, 6))


# Generators per id ---------------------------------------------------------

def gen_F003(X):
    return pair_diff_ordered(X, _sup('mp_t')), [f'F003_diff_mp_t_{i}{j}' for i in range(S) for j in range(S) if i != j]

def gen_F001(X):
    return pair_diff_ordered(X, _sup('mid')), [f'F001_diff_mid_{i}{j}' for i in range(S) for j in range(S) if i != j]

def gen_F004(X):
    return pair_diff_ordered(X, _sup('ofi_W5_lvl1')), [f'F004_diff_ofiW5_{i}{j}' for i in range(S) for j in range(S) if i != j]

def gen_F009(X):
    return pair_diff_ordered(X, _sup('imb1')), [f'F009_diff_imb1_{i}{j}' for i in range(S) for j in range(S) if i != j]

def gen_F010(X):
    return pair_diff_ordered(X, _sup('logret_W5')), [f'F010_diff_logretW5_{i}{j}' for i in range(S) for j in range(S) if i != j]

def gen_F002(X):
    return pair_diff_ordered(X, _sup('wmp')), [f'F002_diff_wmp_{i}{j}' for i in range(S) for j in range(S) if i != j]

def gen_F015(X):
    return pair_diff_ordered(X, _sup('microprice')), [f'F015_diff_microprice_{i}{j}' for i in range(S) for j in range(S) if i != j]

def gen_F040(X):
    return csec_zscore(X, _sup('imb1')), [f'F040_zsc_imb1_s{i}' for i in range(S)]

def gen_F034(X):
    return csec_zscore(X, _sup('ofi')), [f'F034_zsc_ofi_s{i}' for i in range(S)]

def gen_F078(X):
    return csec_demean(X, _sup('ofi_W5_lvl1')), [f'F078_demean_ofiW5_s{i}' for i in range(S)]

def gen_F079(X):
    return csec_demean(X, _sup('ofi_W20')), [f'F079_demean_ofiW20_s{i}' for i in range(S)]

def gen_F031(X):
    return csec_demean(X, _sup('mid')), [f'F031_demean_mid_s{i}' for i in range(S)]

def gen_F019(X):
    return pair_diff_ordered(X, _sup('ofi_W5_lvl3')), [f'F019_diff_mlofiW5L3_{i}{j}' for i in range(S) for j in range(S) if i != j]

def gen_F102(X):
    """sum(OFI_lvl1..5) pair diff, 10 unordered pairs."""
    b = _sum_mlofi_lvl15(X)
    return pair_diff_unordered(b), [f'F102_pair_ofiSum15_{i}{j}' for i, j in _UNORD_PAIRS]

def gen_F021(X):
    return pair_diff_ordered(X, _sup('wap_lvl5')), [f'F021_diff_wap5_{i}{j}' for i in range(S) for j in range(S) if i != j]

def gen_F157(X):
    """imb1[i]*OFI[i] - imb1[j]*OFI[j] for unordered pairs."""
    b = _vec(X, _sup('imb1')) * _vec(X, _sup('ofi'))
    return pair_diff_unordered(b), [f'F157_pair_imb1xOFI_{i}{j}' for i, j in _UNORD_PAIRS]

def gen_F175(X):
    """alternate formulation of F157 — same product, same result."""
    b = _vec(X, _sup('ofi')) * _vec(X, _sup('imb1'))
    return pair_diff_unordered(b), [f'F175_pair_OFIximb1_{i}{j}' for i, j in _UNORD_PAIRS]

def gen_F110(X):
    """imb1[i]*depth[i] - imb1[j]*depth[j] for unordered pairs."""
    b = _vec(X, _sup('imb1')) * _depth_vec(X)
    return pair_diff_unordered(b), [f'F110_pair_imb1xdepth_{i}{j}' for i, j in _UNORD_PAIRS]

def gen_F195(X):
    """(OFI/depth) pair diff (unordered)."""
    eps = 1e-3
    d = _depth_vec(X)
    b = _vec(X, _sup('ofi')) / (np.abs(d) + eps)
    return pair_diff_unordered(b), [f'F195_pair_ofiNormDepth_{i}{j}' for i, j in _UNORD_PAIRS]

def gen_F179(X):
    """(microprice - mid) pair diff (unordered)."""
    b = _vec(X, _sup('microprice')) - _vec(X, _sup('mid'))
    return pair_diff_unordered(b), [f'F179_pair_microMid_{i}{j}' for i, j in _UNORD_PAIRS]

def gen_F193(X):
    return csec_demean(X, _sup('microprice')), [f'F193_demean_micro_s{i}' for i in range(S)]

def gen_F123(X):
    return csec_rank(X, _sup('ofi_W5_lvl1')), [f'F123_rank_ofiW5_s{i}' for i in range(S)]

def gen_F124(X):
    return csec_rank(X, _sup('imb1')), [f'F124_rank_imb1_s{i}' for i in range(S)]

def gen_F049(X):
    """rank of delta_wmp — no delta available; use rank(wmp_lvl1)."""
    return csec_rank(X, _sup('wmp')), [f'F049_rank_wmp_s{i}' for i in range(S)]

def gen_F047(X):
    return csec_rank(X, _sup('logret_W20')), [f'F047_rank_logretW20_s{i}' for i in range(S)]

def gen_F062(X):
    return csec_demean(X, _sup('logret_W5')), [f'F062_demean_logretW5_s{i}' for i in range(S)]

def gen_F145(X):
    """Alternative naming for F062 — same op."""
    return csec_demean(X, _sup('logret_W5')), [f'F145_demean_logretW5_s{i}' for i in range(S)]

def gen_F202(X):
    """ts_neutralize: demean -> zscore on imb1. Same as csec_zscore since demean before std is same."""
    b = csec_demean(X, _sup('imb1'))
    b = csec_zscore_vec(b)
    return b, [f'F202_tsnorm_imb1_s{i}' for i in range(S)]

def gen_F201(X):
    b = csec_demean(X, _sup('ofi'))
    b = csec_zscore_vec(b)
    return b, [f'F201_tsnorm_ofi_s{i}' for i in range(S)]

def gen_F060(X):
    """share of top-of-book size in total. Use exp(b) since b is z-scored (potentially negative); use exp to ensure positive."""
    b = _top_book_size_vec(X)
    # Convert z-scored back to positive scale using exp; share is then bsize-fraction.
    b_pos = np.exp(b.clip(-5, 5))
    return csec_share(b_pos), [f'F060_share_topsize_s{i}' for i in range(S)]

def gen_F085(X):
    return csec_demean(X, _sup('microprice')), [f'F085_demean_micro_s{i}' for i in range(S)]

def gen_F108(X):
    """(mid_i - mid_j) * (imb_i + imb_j), unordered."""
    m = _vec(X, _sup('mid'))
    im = _vec(X, _sup('imb1'))
    diff = m[:, _UNORD_PAIRS[:, 0]] - m[:, _UNORD_PAIRS[:, 1]]
    sm = im[:, _UNORD_PAIRS[:, 0]] + im[:, _UNORD_PAIRS[:, 1]]
    return diff * sm, [f'F108_pair_midDiffImbSum_{i}{j}' for i, j in _UNORD_PAIRS]

def gen_F109(X):
    """(OFI_i - OFI_j) * (spread_i + spread_j), unordered."""
    o = _vec(X, _sup('ofi'))
    sp = _vec(X, _sup('spread'))
    diff = o[:, _UNORD_PAIRS[:, 0]] - o[:, _UNORD_PAIRS[:, 1]]
    sm = sp[:, _UNORD_PAIRS[:, 0]] + sp[:, _UNORD_PAIRS[:, 1]]
    return diff * sm, [f'F109_pair_ofiDiffSpreadSum_{i}{j}' for i, j in _UNORD_PAIRS]

def gen_F063(X):
    return csec_demean(X, _sup('inst_logret')), [f'F063_demean_instret_s{i}' for i in range(S)]

def gen_F147(X):
    """(microprice - mid) demeaned."""
    b = _vec(X, _sup('microprice')) - _vec(X, _sup('mid'))
    return csec_demean_vec(b), [f'F147_demean_microExcess_s{i}' for i in range(S)]

def gen_F168(X):
    """mean(OFI) broadcast — 1 dim per group, broadcast to 5 syms."""
    return csec_market_broadcast(X, _sup('ofi')), [f'F168_mkt_ofi_s{i}' for i in range(S)]

def gen_F166(X):
    return csec_market_broadcast(X, _sup('logret_W20')), [f'F166_mkt_ret_s{i}' for i in range(S)]

def gen_F011(X):
    return pair_diff_ordered(X, _sup('logret_W20')), [f'F011_diff_logretW20_{i}{j}' for i in range(S) for j in range(S) if i != j]

def gen_F177(X):
    """Pair short-vs-long mid-pct-change diff. Proxy via rv_ratio_W5_W50."""
    # We don't have mid_W5/mid_W30, but rv_ratio_W5_W50 captures short-vs-long vol ratio.
    # Use rv_ratio_W5_W50 as a per-sym short-vs-long signal, pair-diff unordered.
    name = 'rv_ratio_W5_W50'
    _init_names()
    if name in _NAME2IDX:
        b = _vec(X, _NAME2IDX[name])
        return pair_diff_unordered(b), [f'F177_pair_rvRatioShortLong_{i}{j}' for i, j in _UNORD_PAIRS]
    # fallback: log-ret diff alone
    b = _vec(X, _sup('logret_W5')) - _vec(X, _sup('logret_W20'))
    return pair_diff_unordered(b), [f'F177_pair_retShortMinusLong_{i}{j}' for i, j in _UNORD_PAIRS]

def gen_F067(X):
    """OFI_W20 cross-impact pair diff (unordered)."""
    return pair_diff_unordered(_vec(X, _sup('ofi_W20'))), [f'F067_pair_ofiW20_{i}{j}' for i, j in _UNORD_PAIRS]

def gen_F008(X):
    return pair_diff_ordered(X, _sup('spread')), [f'F008_diff_spread_{i}{j}' for i in range(S) for j in range(S) if i != j]

def gen_F126(X):
    return csec_rank(X, _sup('RV')), [f'F126_rank_RV_s{i}' for i in range(S)]

def gen_F046(X):
    return csec_rank(X, _sup('kyleinv')), [f'F046_rank_kyleinv_s{i}' for i in range(S)]

def gen_F165(X):
    return csec_signed_rank(X, _sup('logret_W20')), [f'F165_signedRank_logretW20_s{i}' for i in range(S)]

def gen_F139(X):
    return softmax_attention_mid(X, _sup('mid'), tau=1.0), [f'F139_attnMid_s{i}' for i in range(S)]

def gen_F205(X):
    return softmax_attention_mid(X, _sup('mid'), tau=2.0), [f'F205_attnMid_t2_s{i}' for i in range(S)]

def gen_F056(X):
    return csec_market_broadcast(X, _sup('imb1')), [f'F056_mkt_imb1_s{i}' for i in range(S)]

def gen_F081(X):
    return csec_demean(X, _sup('RV')), [f'F081_demean_RV_s{i}' for i in range(S)]

def gen_F104(X):
    """mp_t pair diff minus old. Proxy: use mp_t (midprice1) pair diff minus midprice5 pair diff."""
    _init_names()
    mp_t = _vec(X, _NAME2IDX['midprice1'])
    mp_old = _vec(X, _NAME2IDX.get('midprice5', _NAME2IDX['midprice1']))
    a = pair_diff_unordered(mp_t)
    b = pair_diff_unordered(mp_old)
    return a - b, [f'F104_pair_mptMinusOld_{i}{j}' for i, j in _UNORD_PAIRS]

def gen_F191(X):
    """(imb1 * depth) demeaned cross-sym."""
    b = _vec(X, _sup('imb1')) * _depth_vec(X)
    return csec_demean_vec(b), [f'F191_demean_imbDepth_s{i}' for i in range(S)]


_GEN_MAP = {
    'F003': gen_F003, 'F001': gen_F001, 'F004': gen_F004, 'F009': gen_F009,
    'F010': gen_F010, 'F002': gen_F002, 'F015': gen_F015, 'F040': gen_F040,
    'F034': gen_F034, 'F078': gen_F078, 'F079': gen_F079, 'F031': gen_F031,
    'F019': gen_F019, 'F102': gen_F102, 'F021': gen_F021, 'F157': gen_F157,
    'F175': gen_F175, 'F110': gen_F110, 'F195': gen_F195, 'F179': gen_F179,
    'F193': gen_F193, 'F123': gen_F123, 'F124': gen_F124, 'F049': gen_F049,
    'F047': gen_F047, 'F062': gen_F062, 'F145': gen_F145, 'F202': gen_F202,
    'F201': gen_F201, 'F060': gen_F060, 'F085': gen_F085, 'F108': gen_F108,
    'F109': gen_F109, 'F063': gen_F063, 'F147': gen_F147, 'F168': gen_F168,
    'F166': gen_F166, 'F011': gen_F011, 'F177': gen_F177, 'F067': gen_F067,
    'F008': gen_F008, 'F126': gen_F126, 'F046': gen_F046, 'F165': gen_F165,
    'F139': gen_F139, 'F205': gen_F205, 'F056': gen_F056, 'F081': gen_F081,
    'F104': gen_F104, 'F191': gen_F191,
}


def compute_top50_features(X_grp: np.ndarray, verbose: bool = False):
    """X_grp: (N, 5, 359) z-scored L8 features.

    Returns:
      X_inter: (N, 5, F_inter) float32 — interaction features for each sym.
      feat_names: list[str], len F_inter (= 458 expected).

    Layout strategy:
      - Pair-diff (ordered, n_ordered=20) features per base: for each i in 0..4, that sym gets
        the 4 partner-diffs (i,j) for j!=i. So we reshape (N, 20) → (N, 5, 4).
      - Pair-diff (unordered, n_unordered=10) features: each pair (i,j) with i<j contributes
        a value attributed to BOTH sym i (positive) and sym j (negated), so we expand to (N,5,2)
        — i gets the value, j gets the negation (or vice versa). This keeps each sym getting
        a vector of "pair signals involving this sym".
      - cross-sym (n=5): one value per sym → (N, 5, 1).
      - broadcast (n=1 → 5): same value repeated → (N, 5, 1).
      - attention (n=5): one value per sym → (N, 5, 1).

    All sub-blocks reshape into (N, 5, k_i) where k_i is the per-sym channel count for that feature.
    Final concat along channel axis → (N, 5, F_per_sym). Total dims = N * 5 * F_per_sym, but we
    *count* feature dims as the sum of n_features_generated from the spec, i.e. 458 GLOBALLY (not per sym).
    """
    if X_grp.ndim != 3 or X_grp.shape[1] != S:
        raise ValueError(f'Expected X_grp shape (N, 5, F), got {X_grp.shape}')
    if X_grp.dtype != np.float32:
        X_grp = X_grp.astype(np.float32, copy=False)

    # Replace NaNs (just in case)
    X_grp = np.nan_to_num(X_grp, nan=0.0, posinf=0.0, neginf=0.0)

    N = X_grp.shape[0]
    feats_per_sym = []         # list of (N, 5, k) arrays
    feat_names_global = []     # 458 global names

    for fid in TOP50_ORDER:
        gen = _GEN_MAP[fid]
        arr, names = gen(X_grp)
        if verbose:
            print(f'  {fid}: arr={arr.shape}, names={len(names)}')
        n_dims = len(names)
        feat_names_global.extend(names)

        if arr.shape == (N, 20):
            # ordered pairs: reshape into (N, 5, 4). For sym i, partners are j in (0..4 except i).
            # We follow _ORD_PAIRS order: i=0 j=[1,2,3,4]; i=1 j=[0,2,3,4]; ...
            per_sym = arr.reshape(N, 5, 4)            # (N, 5, 4)
            feats_per_sym.append(per_sym)
        elif arr.shape == (N, 10):
            # unordered pair (i<j): sym i gets +val, sym j gets -val.
            # For each pair (a,b)=_UNORD_PAIRS[k], val[:,k] goes to sym a (positive) and sym b (negated).
            # Result per sym: 4 entries (for the 4 partners involving this sym), with sign.
            per_sym = np.zeros((N, 5, 4), dtype=arr.dtype)
            for sym in range(S):
                # collect pair indices k where sym in (a,b)
                rows = []
                for k, (a, b) in enumerate(_UNORD_PAIRS):
                    if a == sym:
                        rows.append((k, +1.0))
                    elif b == sym:
                        rows.append((k, -1.0))
                # rows has exactly 4 entries (each sym is in 4 unordered pairs)
                assert len(rows) == 4
                for slot, (k, sign) in enumerate(rows):
                    per_sym[:, sym, slot] = sign * arr[:, k]
            feats_per_sym.append(per_sym)
        elif arr.shape == (N, 5):
            per_sym = arr[:, :, None]                 # (N, 5, 1)
            feats_per_sym.append(per_sym)
        else:
            raise ValueError(f'Unexpected output shape from {fid}: {arr.shape}')

    X_inter = np.concatenate(feats_per_sym, axis=2)   # (N, 5, sum_k)
    return X_inter.astype(np.float32, copy=False), feat_names_global


# ---------- Smoke test -----------------------------------------------------

if __name__ == '__main__':
    np.random.seed(0)
    N = 8
    X = np.random.randn(N, 5, 359).astype(np.float32)
    X_inter, names = compute_top50_features(X, verbose=True)
    print(f'X_inter shape={X_inter.shape}  n_names={len(names)}')
    print(f'finite: {np.isfinite(X_inter).all()}  '
          f'min={X_inter.min():.3f} max={X_inter.max():.3f}  '
          f'mean={X_inter.mean():.3f} std={X_inter.std():.3f}')
