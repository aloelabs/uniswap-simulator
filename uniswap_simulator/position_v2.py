import numpy as np

from uniswap_simulator.liquidity_amounts import liquidity_for_amounts, amounts_for_liquidity


class PositionV2:
    def __init__(self, price, fee):
        self._price_sqrt = np.sqrt(price)
        self._fee = fee
        self._gamma = 1. + fee

        self._k = np.zeros_like(price)
        self._x = np.zeros_like(price)
        self._y = np.zeros_like(price)

    def reset(self, price):
        self._price_sqrt = np.sqrt(price)

        self._k = np.zeros_like(price)
        self._x = np.zeros_like(price)
        self._y = np.zeros_like(price)

    @property
    def _price(self):
        return np.square(self._price_sqrt)

    @property
    def fee(self):
        return self._fee

    @property
    def amounts(self):
        return np.vstack((self._x, self._y)).T

    def update(self, price):
        # If price movement is less than fee, it's not guaranteed that the AMM will
        # be arb'd to match new price
        should_update = np.any((
            price / self._price > 1. / (1. - self._fee),
            price / self._price < 1. - self._fee
        ), axis=0)
        price[~should_update] = self._price[~should_update]
        price_sqrt = np.sqrt(price)

        mask = price > self._price # where price is growing

        a = -self._x * (2. + self._fee) + np.sqrt(np.square(self._x * self._fee) + 4 * self._gamma * self._k / price)
        a /= 2. * self._gamma
        b = -self._y * (2. + self._fee) + np.sqrt(np.square(self._y * self._fee) + 4 * self._gamma * self._k * price)
        b /= 2. * self._gamma

        self._x[~mask] += a[~mask]
        self._y[~mask] = price[~mask] * self._x[~mask]
        self._y[mask] += b[mask]
        self._x[mask] = self._y[mask] / price[mask]

        self._x = np.clip(self._x, a_min=0, a_max=None)
        self._y = np.clip(self._y, a_min=0, a_max=None)

        self._k = self._x * self._y
        self._price_sqrt = price_sqrt
        return self.amounts

    def mint(self, amount0, amount1):
        value = np.minimum(amount0 * self._price, amount1)

        self._x += value / self._price
        self._y += value
        self._k = self._x * self._y

        return np.vstack((value / self._price, value)).T
