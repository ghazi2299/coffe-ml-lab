import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config as C

PHASE_CH = ['ph', 'o2', 'co2', 'temp_process', 'temp_scd', 'rh_scd', 'temp_sht', 'rh_sht', 'pressure']


def phase_features(r, win):
    cols = {}
    for c in PHASE_CH:
        x = r[c]
        cols[f'{c}|mean'] = x.rolling(win, min_periods=1, center=True).mean()
        cols[f'{c}|slope'] = x.diff(win).rolling(max(win // 2, 1), min_periods=1, center=True).mean()
    return pd.DataFrame(cols, index=r.index).bfill().ffill()


def relabel(labels):
    order = {}
    for v in labels:
        if v not in order:
            order[v] = len(order)
    return np.array([order[v] for v in labels])


def merge_short(lab, min_len):
    lab = lab.copy()
    changed = True
    while changed:
        changed = False
        brk = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1], True])
        segs = [(brk[i], brk[i + 1]) for i in range(len(brk) - 1)]
        if len(segs) < 2:
            break
        lens = [b - a for a, b in segs]
        i = int(np.argmin(lens))
        if lens[i] >= min_len:
            break
        a, b = segs[i]
        if i == 0:
            lab[a:b] = lab[b]
        elif i == len(segs) - 1:
            lab[a:b] = lab[a - 1]
        else:
            lab[a:b] = lab[a - 1] if lens[i - 1] >= lens[i + 1] else lab[b]
        changed = True
    return lab


def detect_phases(runs, k_range=(2, 5), win=12, smooth=25, min_segment_h=3.0, seed=42):
    feats = {name: phase_features(r, win) for name, r in runs.items()}
    Xall = pd.concat(feats.values())
    scaler = StandardScaler().fit(Xall.values)
    Z = scaler.transform(Xall.values)
    pca = PCA(n_components=0.95, svd_solver='full', random_state=seed).fit(Z)
    Zp = pca.transform(Z)
    best, best_bic = None, np.inf
    for k in range(k_range[0], k_range[1] + 1):
        g = GaussianMixture(k, covariance_type='full', n_init=5, reg_covar=1e-4, random_state=seed).fit(Zp)
        bic = g.bic(Zp)
        if bic < best_bic:
            best, best_bic = g, bic
    out = {}
    for name, f in feats.items():
        lab = best.predict(pca.transform(scaler.transform(f.values)))
        lab = median_filter(lab, size=smooth, mode='nearest')
        out[name] = pd.Series(lab, index=f.index, name='phase')
    allp = relabel(np.concatenate([o.values for o in out.values()]))
    i = 0
    for name in out:
        n = len(out[name])
        out[name] = pd.Series(allp[i:i + n], index=out[name].index, name='phase')
        i += n
    rows = []
    for name in list(out):
        r = runs[name]
        step_h = (r.index[1] - r.index[0]).total_seconds() / 3600.0
        lab = merge_short(out[name].values, int(round(min_segment_h / step_h)))
        out[name] = pd.Series(lab, index=r.index, name='phase')
        brk = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1], True])
        for si in range(len(brk) - 1):
            a, b = brk[si], brk[si + 1]
            rows.append({'run': name, 'segment': si + 1, 'phase': int(lab[a]), 'start': r.index[a],
                         'end': r.index[b - 1], 'hours': float((b - a) * step_h),
                         **{f'{c}_mean': float(r[c].values[a:b].mean()) for c in PHASE_CH}})
    return out, pd.DataFrame(rows), int(best.n_components)
