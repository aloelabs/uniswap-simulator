import numpy as np


class GeometricBrownianMotion:
    def __init__(self, x0, mu, sigma, dt, T):
        self._x0 = x0
        self._mu = mu
        self._sigma = sigma
        self._dt = dt
        self._n = int(T / dt)

    def sample(self, count=1):
        noise = np.random.normal(
            0,
            np.sqrt(self._dt),
            size=(self._n - 1, count)
        )

        x = np.exp((self._mu - 0.5 * self._sigma ** 2) * self._dt + self._sigma * noise)
        x = np.pad(x, ((1, 0), (0, 0)), mode='constant', constant_values=1)

        return self._x0 * x.cumprod(axis=0)
