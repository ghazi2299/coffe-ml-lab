import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.model_selection import RepeatedKFold, LeaveOneOut


def reg_metrics(y, p, thr):
    yc, pc = y >= thr, p >= thr
    out = {
        'MAE': float(mean_absolute_error(y, p)),
        'RMSE': float(np.sqrt(mean_squared_error(y, p))),
        'R2': float(r2_score(y, p)) if len(y) > 1 else np.nan,
        'Spearman': float(spearmanr(y, p).statistic) if len(y) > 2 and np.ptp(p) > 0 else np.nan,
        'Acc_good': float((yc == pc).mean()),
        'F1_good': float(f1_score(yc, pc, zero_division=0)) if yc.any() else np.nan,
    }
    return out


def batch_splits(n, n_splits, n_repeats, seed):
    if n < 10:
        return list(LeaveOneOut().split(np.arange(n))), n
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rkf.split(np.arange(n))), n_splits


def cv_predict(pred_fn, n, splits, folds_per_repeat):
    n_rep = max(1, len(splits) // folds_per_repeat)
    oof = np.full((n_rep, n), np.nan)
    for i, (tr, te) in enumerate(splits):
        oof[i // folds_per_repeat, te] = pred_fn(tr, te)
    return oof


def summarize_oof(y, oof, thr):
    per = [reg_metrics(y, oof[r], thr) for r in range(oof.shape[0])]
    out = {}
    for k in per[0]:
        v = np.array([m[k] for m in per], dtype=float)
        out[k] = float(np.nanmean(v))
        out[k + '_std'] = float(np.nanstd(v))
    return out


def window_fractions(meta):
    run = meta['run'].values
    pos = meta['pos'].values.astype(float)
    frac = np.zeros(len(meta))
    for r in np.unique(run):
        m = run == r
        lo, hi = pos[m].min(), pos[m].max()
        frac[m] = (pos[m] - lo) / max(hi - lo + 1.0, 1.0)
    return frac


def forecast_splits(meta, scheme, n_folds, min_train_frac, maxh, lookback):
    run = meta['run'].values
    pos = meta['pos'].values
    frac = window_fractions(meta)
    runs = np.unique(run)
    out = []
    if scheme == 'loro' and len(runs) >= 3:
        for r in runs:
            te = np.flatnonzero(run == r)
            tr = np.flatnonzero(run != r)
            out.append((tr, te))
        return out
    edges = np.linspace(min_train_frac if scheme == 'rolling' else 0.0, 1.0, n_folds + 1)
    for k in range(n_folds):
        s, e = edges[k], edges[k + 1]
        te_mask = (frac >= s) & (frac < e if k < n_folds - 1 else frac <= e)
        te = np.flatnonzero(te_mask)
        if len(te) == 0:
            continue
        tr_mask = np.zeros(len(meta), dtype=bool)
        for r in runs:
            m = run == r
            pr = pos[m]
            te_pos = pos[m & te_mask]
            if len(te_pos) == 0:
                before = np.ones(m.sum(), dtype=bool)
                tr_mask[np.flatnonzero(m)[before]] = True
                continue
            t0, t1 = te_pos.min(), te_pos.max()
            ok = pr + maxh < t0
            if scheme != 'rolling':
                ok = ok | (pr - lookback + 1 > t1 + maxh)
            tr_mask[np.flatnonzero(m)[ok]] = True
        tr = np.flatnonzero(tr_mask)
        if len(tr) >= 30:
            out.append((tr, te))
    return out


def tail_split(meta, val_frac, maxh):
    run = meta['run'].values
    pos = meta['pos'].values
    fit = np.ones(len(meta), dtype=bool)
    val = np.zeros(len(meta), dtype=bool)
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        pr = pos[idx]
        cut = np.quantile(pr, 1.0 - val_frac)
        v = pr >= cut
        val[idx[v]] = True
        fit[idx[pr + maxh >= cut]] = False
    return np.flatnonzero(fit), np.flatnonzero(val)
