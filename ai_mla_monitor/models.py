import numpy as np


class ScoreColumnModel:
    def __init__(self, column):
        self.column = column

    def fit(self, x, y):
        return self

    def predict_score(self, df):
        return df[self.column].to_numpy(dtype=float)


class LogisticRegressionGD:
    def __init__(self, lr=0.08, epochs=900, l2=0.001):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.mu = None
        self.sigma = None
        self.w = None

    def _scale(self, x):
        return (x - self.mu) / self.sigma

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu = x.mean(axis=0)
        self.sigma = x.std(axis=0)
        self.sigma[self.sigma == 0] = 1.0
        xs = self._scale(x)
        xb = np.c_[np.ones(xs.shape[0]), xs]
        self.w = np.zeros(xb.shape[1])
        for _ in range(self.epochs):
            z = xb @ self.w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
            grad = xb.T @ (p - y) / len(y)
            grad[1:] += self.l2 * self.w[1:]
            self.w -= self.lr * grad
        return self

    def predict_proba(self, x):
        xs = self._scale(np.asarray(x, dtype=float))
        xb = np.c_[np.ones(xs.shape[0]), xs]
        z = xb @ self.w
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class GradientBoostedStumps:
    """Tiny logistic gradient boosting over decision stumps.

    This is intentionally simple and dependency-free. It is a learned nonlinear
    baseline, not a replacement for production tree libraries.
    """

    def __init__(self, n_estimators=60, lr=0.08, candidate_quantiles=9):
        self.n_estimators = n_estimators
        self.lr = lr
        self.candidate_quantiles = candidate_quantiles
        self.init_logit = 0.0
        self.stumps = []

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        p0 = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        self.init_logit = float(np.log(p0 / (1 - p0)))
        f = np.full(len(y), self.init_logit)
        self.stumps = []
        quantiles = np.linspace(0.1, 0.9, self.candidate_quantiles)
        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-np.clip(f, -35, 35)))
            residual = y - p
            best = None
            best_loss = np.inf
            for j in range(x.shape[1]):
                thresholds = np.unique(np.quantile(x[:, j], quantiles))
                for t in thresholds:
                    left = x[:, j] <= t
                    if left.sum() == 0 or (~left).sum() == 0:
                        continue
                    lv = residual[left].mean()
                    rv = residual[~left].mean()
                    pred = np.where(left, lv, rv)
                    loss = np.mean((residual - pred) ** 2)
                    if loss < best_loss:
                        best_loss = loss
                        best = (j, float(t), float(lv), float(rv))
            if best is None:
                break
            j, t, lv, rv = best
            f += self.lr * np.where(x[:, j] <= t, lv, rv)
            self.stumps.append(best)
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        f = np.full(x.shape[0], self.init_logit)
        for j, t, lv, rv in self.stumps:
            f += self.lr * np.where(x[:, j] <= t, lv, rv)
        return 1.0 / (1.0 + np.exp(-np.clip(f, -35, 35)))
