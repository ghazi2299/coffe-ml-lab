import numpy as np
import pandas as pd

from . import config as C
from .windows import INPUTS, motor_events

STAT_CH = list(C.CONTINUOUS) + ['temp_gap', 'temp_gap_sht', 'rh_gap']


def _slope(t, x):
    tc = t - t.mean()
    d = float((tc ** 2).sum())
    return float((tc * (x - x.mean())).sum() / d) if d > 0 else 0.0


def _first_time(t, mask, default):
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if len(idx) else float(default)


def _corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def batch_features(r):
    t = (r.index - r.index[0]).total_seconds().values / 3600.0
    n = len(r)
    dur = float(t[-1]) if n > 1 else 0.0
    dt = dur / max(n - 1, 1)
    k = max(3, n // 12)
    f = {'duration_h': dur}
    for c in STAT_CH:
        x = r[c].values.astype(float)
        parts = np.array_split(x, 3)
        f[f'{c}|mean'] = x.mean()
        f[f'{c}|std'] = x.std()
        f[f'{c}|min'] = x.min()
        f[f'{c}|max'] = x.max()
        f[f'{c}|q10'] = np.quantile(x, 0.1)
        f[f'{c}|q90'] = np.quantile(x, 0.9)
        f[f'{c}|start'] = x[:k].mean()
        f[f'{c}|end'] = x[-k:].mean()
        f[f'{c}|delta'] = x[-k:].mean() - x[:k].mean()
        f[f'{c}|slope'] = _slope(t, x)
        f[f'{c}|early'] = parts[0].mean()
        f[f'{c}|mid'] = parts[1].mean()
        f[f'{c}|late'] = parts[2].mean()
        f[f'{c}|jitter'] = np.diff(x).std() if n > 2 else 0.0
    ph = r['ph'].rolling(5, center=True, min_periods=1).median().values
    dph = np.gradient(ph, t) if n > 2 and dur > 0 else np.zeros(n)
    ph0 = ph[:k].mean()
    for d in (0.1, 0.25, 0.5):
        f[f'ph|t_drop_{d}'] = _first_time(t, ph <= ph0 - d, dur)
    f['ph|hours_below_start_0.2'] = float(dt * np.sum(ph < ph0 - 0.2))
    f['ph|max_drop_rate'] = float(-dph.min())
    f['ph|max_rise_rate'] = float(dph.max())
    f['ph|total_drop'] = float(ph0 - ph.min())
    f['ph|t_min'] = float(t[int(np.argmin(ph))])
    co2 = r['co2'].values.astype(float)
    sat = r['co2_sat'].values
    f['co2|sat_frac'] = float(sat.mean())
    f['co2|t_first_sat'] = _first_time(t, sat > 0.5, dur)
    f['co2|n_jumps'] = float(np.sum(np.abs(np.diff(co2)) > 2500)) if n > 1 else 0.0
    f['co2|gain_6h'] = float(co2[min(n - 1, int(6.0 / max(dt, 1e-9)))] - co2[0])
    f['co2|log_ratio'] = float(np.log((co2.max() + 1.0) / (co2[:k].mean() + 1.0)))
    o2 = r['o2'].rolling(3, center=True, min_periods=1).median().values
    o20 = o2[:k].mean()
    f['o2|drop'] = float(o20 - o2.min())
    f['o2|t_min'] = float(t[int(np.argmin(o2))])
    f['o2|deficit_mean'] = float(np.maximum(o20 - o2, 0).mean())
    tp = r['temp_process'].values.astype(float)
    for base in (25.0, 26.0, 27.0):
        f[f'temp|degree_hours_above_{base:g}'] = float(dt * np.maximum(tp - base, 0).sum())
        f[f'temp|frac_above_{base:g}'] = float((tp > base).mean())
    f['temp|range'] = float(tp.max() - tp.min())
    p = r['pressure'].values.astype(float)
    sd = p.std()
    f['pressure|n_spikes'] = float(np.sum(p > p.mean() + 3 * sd)) if sd > 0 else 0.0
    f['pressure|frac_above_0.05'] = float((p > 0.05).mean())
    f['pressure|lag1_corr'] = _corr(p[:-1], p[1:]) if n > 3 else 0.0
    ev = motor_events(r['motor'].values)
    f['motor|n_events'] = float(len(ev))
    f['motor|mean_interval_h'] = float(np.diff(ev).mean() * dt) if len(ev) > 1 else 0.0
    f['motor|events_per_day'] = float(len(ev) / max(dur / 24.0, 1e-9))
    pairs = [('ph', 'co2'), ('ph', 'o2'), ('o2', 'co2'), ('ph', 'temp_process'),
             ('o2', 'temp_process'), ('co2', 'temp_process'), ('temp_scd', 'temp_process'),
             ('rh_scd', 'rh_sht'), ('pressure', 'o2')]
    for a, b in pairs:
        f[f'corr|{a}~{b}'] = _corr(r[a].values, r[b].values)
    return f


def feature_table(runs):
    rows = {k: batch_features(r) for k, r in runs.items()}
    return pd.DataFrame(rows).T.astype(float)


def sequence_tensor(runs, steps):
    X, S, ids = [], [], []
    grid = np.linspace(0.0, 1.0, steps)
    for k, r in runs.items():
        t = (r.index - r.index[0]).total_seconds().values / 3600.0
        dur = max(float(t[-1]), 1e-6)
        u = t / dur
        ch = [np.interp(grid, u, r[c].values.astype(float)) for c in INPUTS]
        hours = grid * dur
        d_ph = np.gradient(ch[INPUTS.index('ph')], hours)
        d_co2 = np.gradient(ch[INPUTS.index('co2')], hours)
        d_o2 = np.gradient(ch[INPUTS.index('o2')], hours)
        X.append(np.stack(ch + [d_ph, d_co2, d_o2], axis=1))
        ev = motor_events(r['motor'].values)
        S.append([dur, float(len(ev))])
        ids.append(k)
    return np.asarray(X, np.float32), np.asarray(S, np.float32), ids
