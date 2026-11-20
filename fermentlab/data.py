import io
import zipfile
import numpy as np
import pandas as pd

from . import config as C


class DataError(ValueError):
    pass


def _read_one(source):
    df = pd.read_csv(source)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
    missing = [c for c in C.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f'missing columns: {missing}')
    df = df[list(C.RAW_COLUMNS)].rename(columns=C.RAW_COLUMNS)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    return df


def read_logs(paths):
    if isinstance(paths, (str, bytes)) or hasattr(paths, 'read'):
        paths = [paths]
    frames = []
    for p in paths:
        if isinstance(p, str) and p.lower().endswith('.zip'):
            with zipfile.ZipFile(p) as zf:
                for name in sorted(zf.namelist()):
                    if name.lower().endswith('.csv') and '__MACOSX' not in name:
                        frames.append(_read_one(io.BytesIO(zf.read(name))))
        else:
            frames.append(_read_one(p))
    if not frames:
        raise DataError('no readable logs')
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['time']).drop_duplicates(subset='time', keep='last')
    df = df.sort_values('time').reset_index(drop=True)
    if df.empty:
        raise DataError('no valid rows')
    return df


def _hampel(x, window, z):
    med = x.rolling(window, center=True, min_periods=1).median()
    dev = (x - med).abs()
    mad = dev.rolling(window * 4 + 1, center=True, min_periods=1).median() * 1.4826
    scale = np.maximum(mad, 0.25 * dev.median() * 1.4826 + 1e-9)
    return (dev / scale) > z


def clean(df):
    df = df.copy()
    df['co2_sat'] = (pd.to_numeric(df['co2'], errors='coerce') >= C.CO2_SATURATION).astype(float)
    df['motor'] = pd.to_numeric(df['motor'], errors='coerce').fillna(0).clip(0, 1)
    flags = {}
    for c in C.CONTINUOUS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        lo, hi = C.VALID_RANGE[c]
        bad = (df[c] < lo) | (df[c] > hi)
        if c in C.SPIKE_CHANNELS:
            bad = bad | _hampel(df[c].where(~bad), 7, C.SPIKE_CHANNELS[c]).fillna(False)
        df.loc[bad, c] = np.nan
        flags[c] = int(bad.sum())
    df.attrs['glitches'] = flags
    return df


def split_runs(df, gap_min=C.GAP_MINUTES, min_hours=C.MIN_BATCH_HOURS, windows=None):
    if windows is not None:
        w = windows.copy()
        w.columns = [c.strip().lower() for c in w.columns]
        runs = []
        for _, r in w.iterrows():
            m = (df['time'] >= pd.to_datetime(r['start'])) & (df['time'] <= pd.to_datetime(r['end']))
            g = df.loc[m]
            if len(g):
                runs.append((str(r['session']), g.reset_index(drop=True)))
    else:
        bid = (df['time'].diff().dt.total_seconds() > gap_min * 60).cumsum()
        runs = [(str(k + 1), g.reset_index(drop=True)) for k, (_, g) in enumerate(df.groupby(bid))]
    kept = []
    for name, g in runs:
        hours = (g['time'].iloc[-1] - g['time'].iloc[0]).total_seconds() / 3600.0
        if hours >= min_hours:
            kept.append((name, g))
    if not kept:
        raise DataError('no run long enough')
    return dict(kept)


def resample(g, step_min):
    g = g.set_index('time')
    rule = f'{step_min}min'
    agg = {c: 'mean' for c in C.CONTINUOUS}
    agg.update({'co2_sat': 'mean', 'motor': 'max'})
    r = g[list(agg)].resample(rule).agg(agg)
    r[C.CONTINUOUS] = r[C.CONTINUOUS].interpolate(limit=12, limit_direction='both')
    r['co2_sat'] = r['co2_sat'].ffill().bfill()
    r['motor'] = r['motor'].fillna(0)
    r = r.dropna()
    r.index.name = 'time'
    return r


def add_derived(r):
    r = r.copy()
    r['temp_gap'] = r['temp_scd'] - r['temp_process']
    r['temp_gap_sht'] = r['temp_sht'] - r['temp_process']
    r['rh_gap'] = r['rh_scd'] - r['rh_sht']
    return r


def prepare_runs(df, step_min, windows=None, gap_min=C.GAP_MINUTES, min_hours=C.MIN_BATCH_HOURS):
    runs = split_runs(df, gap_min=gap_min, min_hours=min_hours, windows=windows)
    return {k: add_derived(resample(g, step_min)) for k, g in runs.items()}


def load_scores(path):
    s = pd.read_csv(path)
    s.columns = [c.strip().lower() for c in s.columns]
    s = s.dropna(subset=['score'])
    s = s[pd.to_numeric(s['score'], errors='coerce').notna()]
    return pd.Series(s['score'].astype(float).values, index=s['session'].astype(str).values, name='score')
