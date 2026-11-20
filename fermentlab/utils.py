import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import f_regression


class ClippedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, clip=8.0):
        self.clip = clip

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mu_ = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        self.sd_ = np.where(sd < 1e-9, 1.0, sd)
        return self

    def transform(self, X):
        Z = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        return np.clip(Z, -self.clip, self.clip)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.98):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.shape[0] < 3 or X.shape[1] < 2:
            self.keep_ = np.arange(X.shape[1])
            return self
        with np.errstate(invalid='ignore', divide='ignore'):
            R = np.abs(np.corrcoef(X, rowvar=False))
        R = np.nan_to_num(R, nan=0.0)
        keep = []
        for j in range(X.shape[1]):
            if all(R[j, k] < self.threshold for k in keep):
                keep.append(j)
        self.keep_ = np.asarray(keep, dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


class SafeKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=20):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            f, _ = f_regression(X, y)
        f = np.nan_to_num(f, nan=0.0)
        k = min(self.k, X.shape[1])
        self.idx_ = np.argsort(-f)[:k]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.idx_]


class PLS1(PLSRegression):
    def predict(self, X, copy=True):
        return np.ravel(super().predict(X, copy=copy))
