import numpy as np

from uniswap_simulator.position import Position


class CompoundingStrategy(Position):
    epsilon = 0.001

    def compound(self, fraction=0.99):
        if self._earned is None:
            return

        used = self.mint(
            self._earned[..., 0] * fraction,
            self._earned[..., 1] * fraction
        )
        self._earned -= used

        assert self._earned.min() >= -CompoundingStrategy.epsilon, self._earned.min()
        self._earned[self._earned < 0] = 0

    def update(self, price: np.ndarray) -> tuple:
        res = super().update(price)
        self.compound()
        return res
