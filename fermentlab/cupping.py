import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config as C
from . import cv as CV
from . import features as FT
from . import seqnets as SN
from .utils import ClippedScaler, CorrelationFilter, PLS1, SafeKBest


def _pre():
    return [('var', VarianceThreshold(1e-12)),
            ('imp', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('cf', CorrelationFilter(0.98)),
            ('sc', ClippedScaler())]


def _inner(seed):
    return RepeatedKFold(n_splits=4, n_repeats=3, random_state=seed)


def tab_models(cfg):
    rs = cfg.random_state
    out = {}
    out['Ridge'] = Pipeline(_pre() + [('m', RidgeCV(alphas=np.logspace(-2, 4, 40)))])
    sel = Pipeline(_pre() + [('sel', SafeKBest()), ('m', RidgeCV(alphas=np.logspace(-2, 4, 40)))])
    out['RidgeSel'] = GridSearchCV(sel, {'sel__k': [5, 10, 20, 40]}, cv=_inner(rs),
                                   scoring='neg_mean_absolute_error') if cfg.tune else sel
    pls = Pipeline(_pre() + [('m', PLS1(n_components=2, scale=False))])
    out['PLS'] = GridSearchCV(pls, {'m__n_components': [1, 2, 3, 4]}, cv=_inner(rs),
                              scoring='neg_mean_absolute_error', error_score=np.nan) if cfg.tune else pls
    svr = Pipeline(_pre() + [('m', SVR(kernel='rbf'))])
    grid = {'m__C': [0.3, 1.0, 3.0, 10.0, 30.0], 'm__epsilon': [0.05, 0.2], 'm__gamma': ['scale', 'auto']}
    svr_s = GridSearchCV(svr, grid, cv=_inner(rs), scoring='neg_mean_absolute_error') if cfg.tune else svr
    out['SVR'] = TransformedTargetRegressor(regressor=svr_s, transformer=StandardScaler())
    out['RandomForest'] = Pipeline([('var', VarianceThreshold(1e-12)),
                                    ('imp', SimpleImputer(strategy='median', keep_empty_features=True)),
                                    ('m', RandomForestRegressor(n_estimators=cfg.rf_trees, max_features=0.5,
                                                                max_depth=cfg.rf_depth or None,
                                                                min_samples_leaf=1, random_state=rs, n_jobs=-1))])
    out['ExtraTrees'] = Pipeline([('var', VarianceThreshold(1e-12)),
                                  ('imp', SimpleImputer(strategy='median', keep_empty_features=True)),
                                  ('m', ExtraTreesRegressor(n_estimators=cfg.et_trees, max_features=0.5,
                                                            min_samples_leaf=1, random_state=rs, n_jobs=-1))])
    out['GradientBoosting'] = Pipeline([('var', VarianceThreshold(1e-12)),
                                        ('imp', SimpleImputer(strategy='median', keep_empty_features=True)),
                                        ('m', GradientBoostingRegressor(n_estimators=cfg.gb_trees, max_depth=cfg.gb_depth,
                                                                        learning_rate=cfg.gb_lr, subsample=0.8,
                                                                        random_state=rs))])
    kernel = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(0.1)
    out['GP'] = Pipeline(_pre() + [('m', GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                                  n_restarts_optimizer=3, random_state=rs))])
    return {k: v for k, v in out.items() if k in cfg.models}


def embed_model(cfg):
    return Pipeline([('imp', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('sc', ClippedScaler()), ('pca', PCA(n_components=0.95, svd_solver='full')),
                     ('m', RidgeCV(alphas=np.logspace(-2, 4, 40)))])


class SeqBundle:
    def __init__(self, arch, ens, mu, sd, smu, ssd, ym, ys):
        self.arch, self.ens = arch, ens
        self.mu, self.sd, self.smu, self.ssd, self.ym, self.ys = mu, sd, smu, ssd, ym, ys

    def predict(self, X, S):
        p = self.ens.predict((X - self.mu) / self.sd, (S - self.smu) / self.ssd)[:, 0]
        return p * self.ys + self.ym


def fit_seq(arch, X, S, y, cfg, seeds=None):
    mu, sd = X.mean((0, 1)), np.maximum(X.std((0, 1)), 1e-6)
    smu, ssd = S.mean(0), np.maximum(S.std(0), 1e-6)
    ym, ys = float(y.mean()), float(max(y.std(), 1e-6))
    p = dict(epochs=cfg.epochs, patience=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
             batch_size=cfg.batch_size, hidden=cfg.hidden, dropout=cfg.dropout, noise=cfg.noise, crop=0.08)
    Xn = ((X - mu) / sd).astype(np.float32)
    Sn = ((S - smu) / ssd).astype(np.float32)
    yn = ((y - ym) / ys).astype(np.float32)[:, None]
    ens = SN.fit_ensemble(arch, Xn, Sn, yn, p, seeds or cfg.seeds)
    return SeqBundle(arch, ens, mu, sd, smu, ssd, ym, ys)


def evaluate(runs, scores, cfg, forecaster=None, runs_fine=None, progress=None):
    say = progress or (lambda msg: None)
    ids = [k for k in runs if k in scores.index and not pd.isna(scores.loc[k])]
    if len(ids) < 6:
        raise ValueError(f'need at least 6 scored batches, found {len(ids)}')
    runs = {k: runs[k] for k in ids}
    y = scores.loc[ids].values.astype(float)
    n = len(ids)
    F = FT.feature_table(runs)
    X, S, _ = FT.sequence_tensor(runs, cfg.seq_steps)
    E = None
    if cfg.use_embeddings and forecaster is not None and runs_fine is not None:
        E = forecaster.embed_runs({k: runs_fine[k] for k in ids if k in runs_fine})
        if E is not None:
            E = E.reindex(ids)
    splits, fpr = CV.batch_splits(n, cfg.n_splits, cfg.repeats, cfg.random_state)
    thr = cfg.threshold
    oof, rows = {}, []

    def add(approach, name, o):
        oof[name] = o
        rows.append({'Approach': approach, 'Model': name, **CV.summarize_oof(y, o, thr)})

    def sk_fn(model, data):
        def fn(tr, te):
            m = clone(model)
            m.fit(data.iloc[tr], y[tr])
            return np.ravel(m.predict(data.iloc[te]))
        return fn

    say('baseline')
    add('Reference', 'Baseline (mean)', CV.cv_predict(sk_fn(DummyRegressor(strategy='mean'), F), n, splits, fpr))
    models = tab_models(cfg)
    for name, model in models.items():
        say(f'tabular {name}')
        add('Tabular', name, CV.cv_predict(sk_fn(model, F), n, splits, fpr))
    if E is not None and E.notna().any().any():
        say('tabular Ridge+Embedding')
        add('Tabular', 'Ridge with Embedding', CV.cv_predict(sk_fn(embed_model(cfg), E), n, splits, fpr))
    for arch in cfg.archs:
        label = SN.ARCH_LABEL[arch]
        say(f'sequence {label}')

        def fn(tr, te, arch=arch):
            b = fit_seq(arch, X[tr], S[tr], y[tr], cfg)
            return b.predict(X[te], S[te])
        add('Sequence', label, CV.cv_predict(fn, n, splits, fpr))
    res = pd.DataFrame(rows)
    tab = res[res['Approach'] == 'Tabular']
    seq = res[res['Approach'] == 'Sequence']
    best_tab = tab.loc[tab['MAE'].idxmin(), 'Model'] if len(tab) else None
    best_seq = seq.loc[seq['MAE'].idxmin(), 'Model'] if len(seq) else None
    parts = [m for m in (best_tab, best_seq) if m]
    if len(parts) == 2:
        add('Ensemble', 'Ensemble of Best Tabular and Best Sequence', np.mean([oof[m] for m in parts], axis=0))
    res = pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)
    return {'results': res, 'oof': {k: v.mean(0) for k, v in oof.items()}, 'ids': ids, 'y': y, 'F': F,
            'X': X, 'S': S, 'E': E, 'best_tab': best_tab, 'best_seq': best_seq, 'models': models,
            'cv': 'Leave-One-Out' if n < 10 else f'{cfg.n_splits}-fold x {cfg.repeats} repeats'}


class CuppingModel:
    def __init__(self, cfg, tab, seq, tab_name, seq_name, threshold, columns):
        self.cfg, self.tab, self.seq = cfg, tab, seq
        self.tab_name, self.seq_name = tab_name, seq_name
        self.threshold, self.columns = threshold, columns

    def predict(self, runs):
        F = FT.feature_table(runs).reindex(columns=self.columns)
        out = pd.DataFrame(index=pd.Index(list(runs), name='session'))
        cols = []
        if self.tab is not None:
            c = f'{self.tab_name} (Tabular)'
            out[c] = np.ravel(self.tab.predict(F))
            cols.append(c)
        if self.seq is not None:
            X, S, _ = FT.sequence_tensor(runs, self.cfg.seq_steps)
            c = f'{self.seq_name} (Sequence)'
            out[c] = self.seq.predict(X, S)
            cols.append(c)
        out['Average'] = out[cols].mean(axis=1)
        out['Model_gap'] = out[cols].max(axis=1) - out[cols].min(axis=1)
        out['Prediction'] = np.where(out['Average'] >= self.threshold, 'Good', 'Poor')
        return out

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def fit_final(exp, cfg):
    F, y = exp['F'], exp['y']
    tab = clone(exp['models'][exp['best_tab']]).fit(F, y) if exp['best_tab'] in exp['models'] else None
    arch = {v: k for k, v in SN.ARCH_LABEL.items()}.get(exp['best_seq'])
    seq = fit_seq(arch, exp['X'], exp['S'], y, cfg) if arch else None
    return CuppingModel(cfg, tab, seq, exp['best_tab'], exp['best_seq'], cfg.threshold, list(F.columns))


def importance_table(model, F, y, seed=42):
    if model is None:
        return pd.DataFrame()
    r = permutation_importance(model, F, y, scoring='neg_mean_absolute_error', n_repeats=30, random_state=seed)
    sp = [spearmanr(F[c], y).statistic if F[c].nunique() > 1 else np.nan for c in F.columns]
    t = pd.DataFrame({'permutation_importance': r.importances_mean, 'permutation_std': r.importances_std,
                      'spearman_with_score': sp}, index=F.columns)
    return t.sort_values('permutation_importance', ascending=False)


def cluster_batches(F, y, ids, seed=42):
    """Group batches by their sensor-feature profile (k-means, k chosen by silhouette)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = len(ids)
    if n < 4:
        return None
    Z = Pipeline([('imp', SimpleImputer(strategy='median', keep_empty_features=True)),
                  ('var', VarianceThreshold(1e-12)), ('sc', ClippedScaler())]).fit_transform(F)
    xy = PCA(n_components=2, random_state=seed).fit(Z)
    pts = xy.transform(Z)
    best = None
    for k in range(2, min(5, n - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        if len(set(lab)) < 2:
            continue
        sil = float(silhouette_score(Z, lab))
        if best is None or sil > best[0]:
            best = (sil, k, lab)
    if best is None:
        return None
    sil, k, lab = best
    rows = [{'cluster': c + 1, 'batches': int((lab == c).sum()), 'mean_score': float(np.mean(y[lab == c])),
             'min_score': float(np.min(y[lab == c])), 'max_score': float(np.max(y[lab == c]))} for c in range(k)]
    return {'k': k, 'silhouette': sil, 'labels': [int(v) + 1 for v in lab], 'ids': list(ids),
            'x': pts[:, 0].round(3).tolist(), 'y': pts[:, 1].round(3).tolist(), 'summary': rows,
            'explained': [float(v) for v in xy.explained_variance_ratio_]}
