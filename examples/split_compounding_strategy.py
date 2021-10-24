import numpy as np

from uniswap_simulator.position import Position


class SplitCompoundingStrategy():
    epsilon = 0.001

    def __init__(self, price, lower, upper, fee):
        self.position = Position(price, lower, upper, fee)
        self.position_l = Position(price, lower, price, fee)
        self.position_r = Position(price, price, upper, fee)

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
        return amounts

    def _compound(self, price, fraction=0.99):
        if self.position._earned is None:
            return
        
        earned = self.position._earned.copy()
        earned += self.position_l.burn()
        earned += self.position_r.burn()

        edge = price.copy()
        edge[price <= self.position.lower] = self.position.upper
        edge[edge > self.position.upper] = self.position.upper
        self.position_l = Position(
            price,
            self.position.lower,
            edge,
            self.position.fee
        )

        edge = price.copy()
        edge[price >= self.position.upper] = self.position.lower
        edge[edge < self.position.lower] = self.position.lower
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
