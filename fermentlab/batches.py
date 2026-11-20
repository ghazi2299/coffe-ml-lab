import io

import numpy as np
import pandas as pd

from . import config as C
from . import data as D

MIN_BATCHES = 6
TARGET_PSEUDO = 8
MIN_PSEUDO_HOURS = 4.0


def read_cupping(raw):
    """Parse the cupping file. Returns None when empty / unreadable, else a DataFrame (session, score[, start, end])."""
    if not raw:
        return None
    try:
        t = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return None
    t.columns = [str(c).strip().lower().lstrip('\ufeff') for c in t.columns]
    if 'session' not in t.columns or 'score' not in t.columns:
        return None
    t['session'] = t['session'].astype(str).str.strip()
    t['score'] = pd.to_numeric(t['score'], errors='coerce')
    return t.dropna(subset=['session'])


def cupping_problem(raw):
    """Human-readable reason when a cupping file was given but cannot be used; None when fine or absent."""
    if not raw:
        return None
    try:
        t = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return 'File cupping tidak dapat dibaca sebagai file CSV.'
    cols = {str(c).strip().lower().lstrip('\ufeff') for c in t.columns}
    miss = [c for c in ('session', 'score') if c not in cols]
    if miss:
        return ('File cupping kehilangan kolom: ' + ', '.join(miss) +
                '. Minimal butuh kolom session dan score (satu baris per batch).')
    return None


def renumber(runs):
    return {str(i + 1): g for i, g in enumerate(runs.values())}


def pseudo_split(runs, target=TARGET_PSEUDO):
    """Cut every run into contiguous pieces so that there are about `target` sub-batches in total."""
    per = int(np.ceil(target / max(len(runs), 1)))
    out = {}
    for g in runs.values():
        hours = (g.index[-1] - g.index[0]).total_seconds() / 3600.0
        k = max(1, min(per, int(hours // MIN_PSEUDO_HOURS)))
        for part in np.array_split(np.arange(len(g)), k):
            if len(part) >= 4:
                out[str(len(out) + 1)] = g.iloc[part]
    if len(out) < MIN_BATCHES:
        total = sum((g.index[-1] - g.index[0]).total_seconds() for g in runs.values()) / 3600.0
        raise D.DataError(f'Data terlalu pendek untuk membuat {MIN_BATCHES} batch ({total:.0f} jam ditemukan, '
                          f'minimal butuh sekitar {MIN_BATCHES * MIN_PSEUDO_HOURS:.0f} jam).')
    return out


def windows_table(runs, scores):
    return pd.DataFrame([{'session': k, 'start': str(g.index[0]), 'end': str(g.index[-1]),
                          'hours': round((g.index[-1] - g.index[0]).total_seconds() / 3600.0, 1),
                          'score': round(float(scores.get(k)), 1) if k in scores.index else ''}
                         for k, g in runs.items()])


def resolve(df, cup_raw, step_min, seed=0):
    """Return (runs, scores, info). `df` is the cleaned sensor log (same loader as Mode 1)."""
    cup = read_cupping(cup_raw)
    has_windows = cup is not None and {'start', 'end'} <= set(cup.columns)
    if has_windows:
        try:
            runs = D.prepare_runs(df, step_min, windows=cup[['session', 'start', 'end']])
        except Exception:
            runs = D.prepare_runs(df, step_min)
            has_windows = False
    else:
        runs = renumber(D.prepare_runs(df, step_min))
    n_real = len(runs)

    scores, kind = None, None
    if cup is not None:
        s = cup.dropna(subset=['score']).drop_duplicates('session', keep='last')
        s = pd.Series(s['score'].astype(float).values, index=s['session'].values)
        matched = [k for k in runs if k in s.index]
        if len(matched) >= MIN_BATCHES:
            scores, kind = s.loc[matched], 'real'
            runs = {k: runs[k] for k in matched}

    if kind is None:
        rng = np.random.default_rng(seed)
        if n_real >= MIN_BATCHES:
            kind = 'random'
        else:
            kind = 'pseudo'
            runs = pseudo_split(runs)
        scores = pd.Series(rng.uniform(70, 88, len(runs)).round(1), index=list(runs))

    info = {'kind': kind, 'n_real_batches': n_real, 'n_used': len(runs), 'windows_defined': bool(has_windows)}
    return runs, scores, info