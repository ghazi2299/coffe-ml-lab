import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from . import config as C
from . import cv as CV
from . import seqnets as SN
from . import windows as W
from .utils import ClippedScaler


def _flat(a):
    return a.reshape(len(a), -1)


TARGET_CLIP = 4.0


def _alpha(P, Y):
    num = (P * Y).sum(0)
    den = (P * P).sum(0)
    return np.clip(np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0), 0.0, 1.0)


class TabPredictor:
    def __init__(self, est, scale, alpha=None):
        self.est = est
        self.scale = scale
        self.alpha = np.ones(scale.shape) if alpha is None else alpha

    def predict(self, ds):
        p = np.asarray(self.est.predict(ds['tab']))
        p = p.reshape((len(p),) + self.scale.shape)
        return p * self.alpha * self.scale


class SeqPredictor:
    def __init__(self, ens, scaler, scale, alpha=None):
        self.ens = ens
        self.scaler = scaler
        self.scale = scale
        self.alpha = np.ones(scale.shape) if alpha is None else alpha

    def predict(self, ds):
        p = self.ens.predict(self.scaler.transform(ds['seq']), ds['static'])
        p = p.reshape((len(p),) + self.scale.shape)
        return p * self.alpha * self.scale


class MeanPredictor:
    def __init__(self, parts):
        self.parts = parts

    def predict(self, ds):
        return np.mean([p.predict(ds) for p in self.parts], axis=0)


def make_tab(name, cfg, meta, jobs):
    imp = ('imp', SimpleImputer(strategy='median', keep_empty_features=True))
    rs = cfg.random_state
    grid = None
    if name == 'Ridge':
        base = Pipeline([imp, ('sc', ClippedScaler()), ('m', Ridge())])
        grid = {'m__alpha': np.logspace(-1, 4, 11)}
    elif name == 'RandomForest':
        base = Pipeline([imp, ('m', RandomForestRegressor(n_estimators=cfg.rf_trees, max_features=0.33,
                                                           max_depth=cfg.rf_depth or None, min_samples_leaf=3,
                                                           n_jobs=jobs, random_state=rs))])
        grid = {'m__min_samples_leaf': [2, 6]}
    elif name == 'ExtraTrees':
        base = Pipeline([imp, ('m', ExtraTreesRegressor(n_estimators=cfg.et_trees, max_features=0.33,
                                                         min_samples_leaf=3, n_jobs=jobs, random_state=rs))])
        grid = {'m__min_samples_leaf': [2, 6]}
    elif name == 'HistGB':
        h = HistGradientBoostingRegressor(max_iter=cfg.gb_iter, learning_rate=cfg.gb_lr, max_depth=cfg.gb_depth,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=rs)
        base = Pipeline([imp, ('m', MultiOutputRegressor(h, n_jobs=jobs))])
    else:
        raise ValueError(f'unknown tabular model {name}')
    if cfg.tune and grid is not None:
        splits = CV.forecast_splits(meta, 'rolling', 4, 0.4, max(cfg.horizon_steps), cfg.lookback_steps)
        if len(splits) >= 2:
            return GridSearchCV(base, grid, cv=splits, scoring='neg_mean_squared_error', refit=True)
    return base


def fit_tab(name, dtr, cfg, scale, jobs=-1):
    Y = np.clip(dtr['Y'] / scale, -TARGET_CLIP, TARGET_CLIP)
    alpha = None
    fi, ci = CV.tail_split(dtr['meta'], cfg.val_frac, max(cfg.horizon_steps))
    if cfg.calibrate and len(ci) >= 20 and len(fi) >= 60:
        sub = W.subset(dtr, fi)
        est0 = make_tab(name, cfg, sub['meta'], jobs)
        est0.fit(sub['tab'], _flat(Y[fi]))
        P = np.asarray(est0.predict(dtr['tab'].iloc[ci])).reshape((len(ci),) + scale.shape)
        alpha = _alpha(P, Y[ci])
    est = make_tab(name, cfg, dtr['meta'], jobs)
    est.fit(dtr['tab'], _flat(Y))
    return TabPredictor(est, scale, alpha)


def fit_seq(arch, dtr, cfg, scale, device=None):
    sc = W.SeqScaler().fit(dtr['seq'])
    X = sc.transform(dtr['seq'])
    S = dtr['static']
    Yc = np.clip(dtr['Y'] / scale, -TARGET_CLIP, TARGET_CLIP)
    Y = _flat(Yc)
    fi, vi = CV.tail_split(dtr['meta'], cfg.val_frac, max(cfg.horizon_steps))
    p = dict(epochs=cfg.epochs, patience=cfg.patience, lr=cfg.lr, weight_decay=cfg.weight_decay,
             batch_size=cfg.batch_size, hidden=cfg.hidden, dropout=cfg.dropout, noise=cfg.noise, crop=0.0)
    val = None
    if len(vi) >= 20 and len(fi) >= 60:
        val = (X[vi], S[vi], Y[vi])
    else:
        fi = np.arange(len(X))
    ens = SN.fit_ensemble(arch, X[fi], S[fi], Y[fi], p, cfg.seeds, val, device)
    alpha = None
    if val is not None and cfg.calibrate:
        P = ens.predict(X[vi], S[vi]).reshape((len(vi),) + scale.shape)
        alpha = _alpha(P, Yc[vi])
    return SeqPredictor(ens, sc, scale, alpha)


def cell_metrics(P, Yd, roll_d):
    err = P - Yd
    mae = np.abs(err).mean(0)
    mae_p = np.abs(Yd).mean(0)
    mae_r = np.abs(roll_d - Yd).mean(0)
    rmse = np.sqrt((err ** 2).mean(0))
    ref = np.maximum(Yd.std(0), 1e-9)
    sst = ((Yd - Yd.mean(0)) ** 2).sum(0)
    r2 = 1.0 - (err ** 2).sum(0) / np.maximum(sst, 1e-12)
    pc = P - P.mean(0)
    yc = Yd - Yd.mean(0)
    den = np.sqrt((pc ** 2).sum(0) * (yc ** 2).sum(0))
    corr = np.where(den > 1e-12, (pc * yc).sum(0) / np.maximum(den, 1e-12), np.nan)
    return {
        'MAE': mae, 'RMSE': rmse, 'nMAE': mae / ref, 'R2_delta': r2, 'Corr': corr,
        'Skill_persist': 1.0 - mae / np.maximum(mae_p, 1e-12),
        'Skill_roll': 1.0 - mae / np.maximum(mae_r, 1e-12),
    }


def detail_table(name, m, cfg):
    rows = []
    for i, ch in enumerate(W.TARGETS):
        for j, hm in enumerate(cfg.horizons_min):
            rows.append({'Model': name, 'Channel': ch, 'Horizon_min': hm,
                         **{k: float(v[i, j]) for k, v in m.items()}})
    return pd.DataFrame(rows)


def summarize(detail):
    g = detail.groupby('Model')
    s = g.agg(nMAE=('nMAE', 'mean'), R2_delta=('R2_delta', 'mean'), Corr=('Corr', 'mean'),
              Skill_persist=('Skill_persist', 'mean'), Skill_roll=('Skill_roll', 'mean'))
    s['Skill_persist_median'] = g['Skill_persist'].median()
    s['Cells_beat_persist'] = g['Skill_persist'].apply(lambda v: float((v > 0).mean()))
    return s.sort_values('nMAE')


BT_WINDOWS = 80


def _nested(a):
    a = np.round(np.asarray(a, dtype=float), 5)
    o = a.astype(object)
    o[~np.isfinite(a)] = None
    return o.tolist()


def pack_backtest(meta, last, Y, preds, cfg, max_windows=BT_WINDOWS):
    """Actual and predicted values (absolute units) for evenly spaced windows.

    preds maps a model name to its predicted change, shaped (windows, channels, horizons).
    """
    n = len(meta)
    idx = np.unique(np.linspace(0, n - 1, min(n, max_windows)).astype(int)) if n else np.array([], dtype=int)
    base = last[idx][:, :, None]
    return {
        'time': [t.strftime('%Y-%m-%d %H:%M') for t in meta['time'].iloc[idx]],
        'run': [str(r) for r in meta['run'].iloc[idx]],
        'channels': list(W.TARGETS),
        'horizons': [int(h) for h in cfg.horizons_min],
        'actual': _nested(base + Y[idx]),
        'pred': {k: _nested(base + v[idx]) for k, v in preds.items()},
    }


def fold_table(meta, splits):
    rows = []
    for k, (tr, te) in enumerate(splits):
        t = meta['time'].iloc[te]
        rows.append({'fold': k + 1, 'train_windows': int(len(tr)), 'test_windows': int(len(te)),
                     'test_start': t.min().strftime('%Y-%m-%d %H:%M'), 'test_end': t.max().strftime('%Y-%m-%d %H:%M')})
    return pd.DataFrame(rows)


def run_experiment(runs, cfg, progress=None):
    say = progress or (lambda msg: None)
    period = W.estimate_period(runs)
    ds = W.assemble(runs, cfg, period)
    meta = ds['meta']
    H = max(cfg.horizon_steps)
    splits = CV.forecast_splits(meta, cfg.scheme, cfg.n_folds, cfg.min_train_frac, H, cfg.lookback_steps)
    if not splits:
        raise ValueError('not enough windows for validation')
    N = len(meta)
    T, nh = ds['Y'].shape[1], ds['Y'].shape[2]
    tab_names = list(cfg.tab_models)
    seq_names = {SN.ARCH_LABEL[a]: a for a in cfg.archs}
    oof = {n: np.full((N, T, nh), np.nan) for n in tab_names + list(seq_names)}
    for k, (tr, te) in enumerate(splits):
        dtr, dte = W.subset(ds, tr), W.subset(ds, te)
        scale = W.target_scale(dtr['Y'], dtr['last'].std(0))
        for n in tab_names:
            say(f'fold {k + 1}/{len(splits)}  {n}')
            oof[n][te] = fit_tab(n, dtr, cfg, scale).predict(dte)
        for label, arch in seq_names.items():
            say(f'fold {k + 1}/{len(splits)}  {label}')
            oof[label][te] = fit_seq(arch, dtr, cfg, scale).predict(dte)
    tested = np.zeros(N, dtype=bool)
    for te in [s[1] for s in splits]:
        tested[te] = True
    Yd = ds['Y'][tested]
    roll_d = (ds['roll'] - ds['last'])[tested][:, :, None] * np.ones((1, 1, nh))
    details = [
        detail_table('Persistence', cell_metrics(np.zeros_like(Yd), Yd, roll_d), cfg),
        detail_table('RollingMean12', cell_metrics(roll_d, Yd, roll_d), cfg),
    ]
    for n, o in oof.items():
        details.append(detail_table(n, cell_metrics(o[tested], Yd, roll_d), cfg))
    detail = pd.concat(details, ignore_index=True)
    summary = summarize(detail)
    tab_rank = [n for n in summary.index if n in tab_names]
    seq_rank = [n for n in summary.index if n in seq_names]
    best_tab = tab_rank[0] if tab_rank else None
    best_seq = seq_rank[0] if seq_rank else None
    parts = [n for n in (best_tab, best_seq) if n]
    preds = {'Persistence': np.zeros_like(Yd), **{n: o[tested] for n, o in oof.items()}}
    if len(parts) == 2:
        ens = np.mean([oof[n][tested] for n in parts], axis=0)
        preds['Ensemble'] = ens
        detail = pd.concat([detail, detail_table('Ensemble', cell_metrics(ens, Yd, roll_d), cfg)], ignore_index=True)
        summary = summarize(detail)
    say('fitting final models on all windows')
    scale = W.target_scale(ds['Y'], ds['last'].std(0))
    final = {}
    if best_tab:
        final[best_tab] = fit_tab(best_tab, ds, cfg, scale)
    if best_seq:
        final[best_seq] = fit_seq(seq_names[best_seq], ds, cfg, scale)
    if len(final) == 2:
        final['Ensemble'] = MeanPredictor(list(final.values()))
    default = 'Ensemble' if 'Ensemble' in final else (best_tab or best_seq)
    model = ForecastModel(cfg, period, final, default, best_seq)
    backtest = pack_backtest(meta[tested].reset_index(drop=True), ds['last'][tested], Yd, preds, cfg)
    return {'model': model, 'summary': summary, 'detail': detail, 'splits': len(splits),
            'windows': N, 'tested': int(tested.sum()), 'period_steps': period, 'oof': oof,
            'meta': meta, 'best_tab': best_tab, 'best_seq': best_seq,
            'backtest': backtest, 'folds': fold_table(meta, splits)}


class ForecastModel:
    def __init__(self, cfg, period, predictors, default, seq_name):
        self.cfg = cfg
        self.period = period
        self.predictors = predictors
        self.default = default
        self.seq_name = seq_name

    def predict_tail(self, runs, which=None):
        ds = W.assemble(runs, self.cfg, self.period, tail=True)
        delta = self.predictors[which or self.default].predict(ds)
        rows = []
        for i, run in enumerate(ds['meta']['run']):
            t0 = ds['meta']['time'].iloc[i]
            for j, hm in enumerate(self.cfg.horizons_min):
                row = {'run': run, 'forecast_time': t0 + pd.Timedelta(minutes=int(hm)), 'horizon_min': hm}
                for c, ch in enumerate(W.TARGETS):
                    row[ch] = float(ds['last'][i, c] + delta[i, c, j])
                rows.append(row)
        return pd.DataFrame(rows)

    def backtest(self, runs, which=None, max_windows=120):
        """Compare predictions with what really happened, for every window of the runs that has a known future."""
        ds = W.assemble(runs, self.cfg, self.period)
        name = which if which in self.predictors else self.default
        delta = self.predictors[name].predict(ds)
        out = pack_backtest(ds['meta'], ds['last'], ds['Y'], {name: delta, 'Persistence': np.zeros_like(delta)},
                            self.cfg, max_windows)
        out['model'] = name
        out['windows'] = int(len(ds['meta']))
        return out

    def embed_runs(self, runs):
        pred = self.predictors.get(self.seq_name)
        if pred is None:
            return None
        ds = W.assemble(runs, self.cfg, self.period)
        E = pred.ens.embed(pred.scaler.transform(ds['seq']))
        E = pd.DataFrame(E, index=ds['meta']['run'].values)
        return E.groupby(level=0).mean().reindex(list(runs))

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
