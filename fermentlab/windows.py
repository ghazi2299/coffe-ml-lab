import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import config as C

INPUTS = list(C.CONTINUOUS) + ['temp_gap', 'temp_gap_sht', 'rh_gap', 'co2_sat']
TARGETS = list(C.CONTINUOUS)
N_IN = len(INPUTS)
N_EXTRA = 3
N_SEQ_CH = 2 * N_IN + N_EXTRA


def motor_events(motor):
    m = (np.asarray(motor) > 0.5).astype(int)
    return np.flatnonzero((m == 1) & (np.r_[0, m[:-1]] == 0))


def estimate_period(runs):
    d = []
    for r in runs.values():
        e = motor_events(r['motor'].values)
        if len(e) > 1:
            d.extend(np.diff(e).tolist())
    if not d:
        return None
    p = float(np.median(d))
    return p if p >= 2 else None


def schedule_matrix(motor, period, hsteps):
    n = len(motor)
    out = np.zeros((n, 2 + len(hsteps)))
    if period is None:
        return out
    ev = motor_events(motor)
    idx = np.arange(n)
    if len(ev):
        pos = np.searchsorted(ev, idx, side='right') - 1
        last = np.where(pos >= 0, ev[np.maximum(pos, 0)], ev[0] - period * np.ceil((ev[0] - idx) / period))
        since = (idx - last) % period
    else:
        since = np.zeros(n)
    out[:, 0] = since / period
    out[:, 1] = (period - since) / period
    for j, h in enumerate(hsteps):
        out[:, 2 + j] = np.floor((since + h) / period)
    return out


def hod_matrix(index):
    h = (index.hour.values + index.minute.values / 60.0) / 24.0
    return np.column_stack([np.sin(2 * np.pi * h), np.cos(2 * np.pi * h)])


def _slope_weights(w):
    t = np.arange(w) - (w - 1) / 2.0
    return t / (t ** 2).sum()


def sched_names(hsteps):
    return ['sch_since', 'sch_to_next'] + [f'sch_n_{h}' for h in hsteps]


def tab_frame(r, cfg, sched, hod):
    L = cfg.lookback_steps
    lags = [k for k in (1, 3, 6, 12, 24, 48, 72) if k < L]
    wins = [w for w in (6, 24, 72) if w <= L]
    ext = [w for w in (24, 72) if w <= L]
    slw = [w for w in (12, 36) if w <= L]
    cols = {}
    for c in INPUTS:
        x = r[c]
        cols[f'{c}|last'] = x
        for k in lags:
            cols[f'{c}|d{k}'] = x - x.shift(k)
        for w in wins:
            m = x.rolling(w, min_periods=w)
            cols[f'{c}|rm{w}'] = x - m.mean()
            cols[f'{c}|sd{w}'] = m.std()
        for w in ext:
            m = x.rolling(w, min_periods=w)
            cols[f'{c}|hi{w}'] = m.max() - x
            cols[f'{c}|lo{w}'] = x - m.min()
        for s in (6, 24):
            cols[f'{c}|ew{s}'] = x - x.ewm(span=s, adjust=False).mean()
        for w in slw:
            wt = _slope_weights(w)
            cols[f'{c}|sl{w}'] = x.rolling(w, min_periods=w).apply(lambda v, wt=wt: float(np.dot(wt, v)), raw=True)
    df = pd.DataFrame(cols, index=r.index)
    names = sched_names(cfg.horizon_steps)
    for j, nme in enumerate(names):
        df[nme] = sched[:, j]
    df['hod_sin'] = hod[:, 0]
    df['hod_cos'] = hod[:, 1]
    return df


def run_arrays(r, cfg, period):
    sched = schedule_matrix(r['motor'].values, period, cfg.horizon_steps)
    hod = hod_matrix(r.index)
    tab = tab_frame(r, cfg, sched, hod)
    base = np.column_stack([r[c].values for c in INPUTS] + [r['motor'].values, hod])
    tv = np.column_stack([r[c].values for c in TARGETS])
    roll = np.column_stack([r[c].rolling(12, min_periods=1).mean().values for c in TARGETS])
    return {'tab': tab, 'sched': sched, 'base': base, 'tv': tv, 'roll': roll, 'n': len(r)}


def assemble(runs, cfg, period, tail=False):
    L, hs = cfg.lookback_steps, cfg.horizon_steps
    H = max(hs)
    tabs, seqs, stat, Ys, lasts, rolls, metas = [], [], [], [], [], [], []
    for name, r in runs.items():
        a = run_arrays(r, cfg, period)
        n = a['n']
        ends = np.array([n - 1]) if tail else np.arange(L - 1, n - H)
        if len(ends) == 0 or n < L:
            continue
        win = np.transpose(sliding_window_view(a['base'], L, axis=0), (0, 2, 1))
        seqs.append(win[ends - (L - 1)])
        tabs.append(a['tab'].iloc[ends])
        stat.append(a['sched'][ends])
        lasts.append(a['tv'][ends])
        rolls.append(a['roll'][ends])
        if not tail:
            Ys.append(np.stack([a['tv'][ends + h] - a['tv'][ends] for h in hs], axis=2))
        metas.append(pd.DataFrame({'run': name, 'pos': ends, 'time': r.index[ends]}))
    if not seqs:
        raise ValueError('no windows could be built; runs are shorter than lookback + max horizon')
    ds = {
        'tab': pd.concat(tabs, ignore_index=True),
        'seq': np.concatenate(seqs).astype(np.float32),
        'static': np.concatenate(stat).astype(np.float32),
        'last': np.concatenate(lasts),
        'roll': np.concatenate(rolls),
        'meta': pd.concat(metas, ignore_index=True),
    }
    if not tail:
        ds['Y'] = np.concatenate(Ys)
    return ds


def subset(ds, idx):
    out = {}
    for k, v in ds.items():
        if k == 'tab' or k == 'meta':
            out[k] = v.iloc[idx].reset_index(drop=True)
        else:
            out[k] = v[idx]
    return out


class SeqScaler:
    def fit(self, seq):
        x = seq[:, :, :N_IN].reshape(-1, N_IN)
        self.mu = x.mean(0)
        sd = x.std(0)
        self.sd = np.maximum(sd, 1e-3 + 0.01 * np.abs(self.mu))
        return self

    def transform(self, seq):
        x = seq[:, :, :N_IN]
        lev = (x - self.mu) / self.sd
        rel = (x - x[:, -1:, :]) / self.sd
        out = np.concatenate([lev, rel, seq[:, :, N_IN:]], axis=2)
        return np.clip(out, -10.0, 10.0).astype(np.float32)


def target_scale(Y, last_std):
    sd = Y.std(0)
    floor = 0.01 * last_std[:, None] + 1e-6
    return np.maximum(sd, floor)
