# LightGBM custom multiclass objective — implementation cookbook

Sources:
* Hippocampus's Garden: https://hippocampus-garden.com/lgbm_custom/
* Max Halford "Focal loss implementation for LightGBM": https://maxhalford.github.io/blog/lightgbm-focal-loss/
* Nima Sarang: https://nimasarang.com/blog/2024-08-11-gbt-custom-loss/
* Luca Massaron Kaggle: https://www.kaggle.com/code/lucamassaron/lightgbm-with-multiclass-focal-loss
* jrzaurin/LightGBM-with-Focal-Loss (GitHub)
* y2019xcj/focalloss-for-lightgbm-xgboost (GitHub)

## API contract

LightGBM custom multiclass objective signature:

```python
def fobj(z_flat, dataset):
    """
    z_flat: 1D ndarray of length N*K, F-ordered
            (so z_flat[k*N + i] = logit_k_for_sample_i)
    dataset: lightgbm.Dataset
        - dataset.label: (N,) class indices
        - dataset.weight: (N,) optional sample weights
    Returns:
        grad: 1D ndarray of length N*K, F-ordered
        hess: 1D ndarray of length N*K, F-ordered (must be > 0)
    """
    return grad, hess

def feval(z_flat, dataset):
    """Same signature, returns (name, value, is_higher_better)."""
    return 'metric_name', metric_val, True
```

**Key gotchas**:
1. `z_flat` is **F-ordered**, NOT C-ordered. Always `reshape(N, K, order='F')`.
2. The model outputs **raw logits**, not probabilities. Apply softmax in fobj.
3. Hessian must be **strictly positive** (LightGBM uses Newton step `δ = -g/h`).
   If your true hessian goes negative, clip with `np.maximum(hess, 1e-6)`.
4. `init_score` matters a lot. The first round predicts 0 logits → softmax
   uniform → gradient blows up if reward magnitudes are extreme. Solution:
   warm-start with CE (50 rounds) and pass `init_score=ce_logits.flatten('F')`
   to the new Dataset.

## Template (multiclass softmax with arbitrary per-sample reward vector)

```python
import numpy as np
import lightgbm as lgb

def make_pnl_obj(rewards_func, K=3, beta_entropy=0.0):
    """
    rewards_func(dataset) -> ndarray (N, K) of per-sample per-class rewards.
    Returns an objective that minimises -E[reward].
    """
    def obj(z_flat, dataset):
        N = len(dataset.label)
        z = z_flat.reshape(N, K, order='F')
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        p = exp_z / exp_z.sum(axis=1, keepdims=True)
        r = rewards_func(dataset)               # (N, K)
        E_r = (p * r).sum(axis=1, keepdims=True)
        # ∂(-E[r])/∂z_k = -p_k(r_k - E[r])
        grad = -p * (r - E_r)
        # entropy reg: ∂(-βH)/∂z_k = β·p_k·(log p_k - sum_j p_j log p_j)
        if beta_entropy > 0:
            log_p = np.log(p + 1e-12)
            H = (p * log_p).sum(axis=1, keepdims=True)
            grad += beta_entropy * p * (log_p - H)
        # diagonal Newton hessian
        hess = p * (1 - p) * np.maximum(np.abs(r - E_r), 1e-3)
        if beta_entropy > 0:
            hess += beta_entropy * p * (1 - p)
        if dataset.get_weight() is not None:
            w = dataset.get_weight().reshape(-1, 1)
            grad *= w
            hess *= w
        return grad.flatten('F'), hess.flatten('F')
    return obj


def make_pnl_eval(K=3, fee=1e-4):
    def eval_(z_flat, dataset):
        N = len(dataset.label)
        z = z_flat.reshape(N, K, order='F')
        pred = z.argmax(axis=1)
        delta = dataset.delta            # custom attribute
        pnl = np.where(pred == 2, delta - 2*fee,
              np.where(pred == 0, -delta - 2*fee, 0.0)).sum()
        return 'cum_pnl', float(pnl), True
    return eval_


# ----------- usage -----------
fee = 1e-4
def rewards(dataset):
    delta = dataset.delta
    N = len(delta)
    return np.stack([-delta - 2*fee, np.zeros(N), +delta - 2*fee], axis=1)

# Phase 1: warm-start with CE
ds_warm = lgb.Dataset(X_train, label=y_train, weight=w_train)
booster_ce = lgb.train(
    {'objective': 'multiclass', 'num_class': 3, 'learning_rate': 0.05, 'metric': 'multi_logloss'},
    ds_warm, num_boost_round=50,
)
init_score = booster_ce.predict(X_train, raw_score=True)   # (N, 3)

# Phase 2: PnL fine-tune with custom objective
ds = lgb.Dataset(X_train, label=y_train, weight=w_train,
                 init_score=init_score.flatten('F'))
ds.delta = delta_train

ds_val = lgb.Dataset(X_val, label=y_val,
                     init_score=booster_ce.predict(X_val, raw_score=True).flatten('F'),
                     reference=ds)
ds_val.delta = delta_val

obj = make_pnl_obj(rewards, K=3, beta_entropy=0.005)
ev  = make_pnl_eval(K=3, fee=fee)

booster = lgb.train(
    {'learning_rate': 0.02, 'verbose': -1, 'num_class': 3,
     'min_data_in_leaf': 500, 'lambda_l2': 1.0},
    ds,
    num_boost_round=2000,
    valid_sets=[ds_val],
    fobj=obj,
    feval=ev,
    callbacks=[lgb.early_stopping(50)],
)
```

## Why warm-start with CE

Without warm-start the gradient at 0 logits has the form
`p = [1/3, 1/3, 1/3]; E[r] = (r_0 + r_2 - 4f)/3 ≈ -2f/3 (since r_0 ≈ -r_2)`,
so `grad_0 = -1/3·(r_0 + 2f/3) ≈ +Δp/9` etc. — the model is asked to learn
direction without any prior. Result: many rounds spent re-discovering CE-level
features. CE warm-start saves 100+ rounds.

## Sanity checks before training

1. **`np.isfinite(grad).all()`** at first round — if not, your softmax is
   numerically blowing up; check init_score scale.
2. **`hess.min() > 0`** — LightGBM will fail silently if not.
3. After round 1, **`(p == 1.0).any()`** should be False — collapse to one
   class kills the gradient.
4. **Track `cum_pnl` (custom feval)** every 5 rounds. Should monotonically
   increase on val for ≥50 rounds.

## Composing with sample weights (S1 inside S2)

`weight=|Δp|`-weighted samples can be passed via `dataset.weight`. The
gradient inside `obj` already multiplies by w in the snippet above. So
**S1+S2 stack naturally**.

## Multiclass hessian: closed-form vs diagonal

The full softmax+expected-reward hessian is non-diagonal. LightGBM accepts
only diagonal hessians. The diagonal-Newton trick `h_k ≈ p_k(1-p_k)|r_k - E[r]|`
is what every multiclass focal-loss implementation uses. It's not exact but
empirically converges within 5-10% of full Newton.
