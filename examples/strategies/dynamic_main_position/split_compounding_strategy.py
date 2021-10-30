import numpy as np

from uniswap_simulator import Position


class SplitCompoundingStrategy():
    epsilon = 0.001

    def __init__(self, price, lower, upper, fee):
        self.position = Position(price, lower, upper, fee)
        self.position_l = Position(price, lower, price, fee)
        self.position_r = Position(price, price, upper, fee)

        self.half_width = (np.log(upper) - np.log(lower)) / \
            (2 * np.log(1.0001))

    def reset(self, price):
        self.position.reset(price)
        self.position_l = Position(price, self.position.lower, price, self.position.fee)
        self.position_r = Position(price, price, self.position.upper, self.position.fee)

    def mint(self, amount0, amount1):
        return self.position.mint(amount0, amount1)

    def update(self, price):
        amounts = self.position.update(price)
        amounts += self.position_l.update(price)
        amounts += self.position_r.update(price)

        self._compound(price)

        center = np.log(price) / np.log(1.0001)
        lower = np.zeros_like(center)
        upper = np.zeros_like(center)
        mask = center < 0.0
        lower[mask] = np.clip(center - self.half_width,
                              a_min=-887272, a_max=None)[mask]
        upper[mask] = (lower + 2 * self.half_width)[mask]
        upper[~mask] = np.clip(center + self.half_width,
                               a_min=None, a_max=+887272)[~mask]
        lower[~mask] = (upper - 2 * self.half_width)[~mask]
        lower = np.power(1.0001, lower)
        upper = np.power(1.0001, upper)

        burned = self.position.burn()
        self.position = Position(price, lower, upper, self.position.fee)
        used = self.position.mint(burned[..., 0], burned[..., 1])
        self.position._earned = np.clip(burned - used, a_min=0, a_max=None)

        return amounts

    def _compound(self, price, fraction=0.99):
        earned = self.position._earned.copy()
        earned += self.position_l.burn()
        earned += self.position_r.burn()

        edge = price.copy()
        mask = price <= self.position.lower
        edge[mask] = self.position.upper[mask]
        mask = edge > self.position.upper
        edge[mask] = self.position.upper[mask]
        self.position_l = Position(
            price,
            self.position.lower,
            edge,
            self.position.fee
        )

        edge = price.copy()
        mask = price >= self.position.upper
        edge[mask] = self.position.lower[mask]
        mask = edge < self.position.lower
        edge[mask] = self.position.lower[mask]
        self.position_r = Position(
            price,
            edge,
            self.position.upper,
            self.position.fee
        )

        used = self.position_l.mint(np.zeros_like(earned[...,1]), earned[...,1] * fraction)
        used += self.position_r.mint(earned[...,0] * fraction, np.zeros_like(earned[...,0]))

        self.position._earned = earned - used
        assert self.position._earned.min() >= -SplitCompoundingStrategy.epsilon, self.position._earned.min()
        self.position._earned[self.position._earned < 0] = 0
