import numpy as np

from uniswap_simulator import Position


class LiquiditySilos():

    def __init__(self, price, lower, upper, fee):
        self.position = Position(price, lower, upper, fee)
        self.half_width = (np.log(upper) - np.log(lower)) / \
            (2 * np.log(1.0001))

        self.portion_in_uni = 1.0 - np.power(1.0001, -self.half_width / 2.0)
        print('Uniswap %: {:.3f}'.format(self.portion_in_uni.mean()))

        self.silos = None

    def reset(self, price):
        self.position.reset(price)

    def mint(self, amount0, amount1):
        in_uniswap = self.position.mint(
            amount0 * self.portion_in_uni,
            amount1 * self.portion_in_uni
        )
        in_silos = np.vstack((amount0, amount1)).T - in_uniswap
        assert in_silos.min() > 0, in_silos.min()

        if self.silos is None:
            self.silos = in_silos
        else:
            self.silos += in_silos

        return in_uniswap + in_silos

    def update(self, price):
        amounts = self.position.update(price) + self.silos

        center = np.log(price) / np.log(1.0001)
        lower = np.zeros_like(center)
        upper = np.zeros_like(center)

        mask = np.any((center < lower, center > upper), axis=0)
        lower[mask] = (center - self.half_width)[mask]
        upper[mask] = (center + self.half_width)[mask]
        lower = np.power(1.0001, lower)
        upper = np.power(1.0001, upper)

        burned = self.position.burn_at(mask)
        self.position._lower_sqrt[mask] = np.sqrt(lower[mask])
        self.position._upper_sqrt[mask] = np.sqrt(upper[mask])

        to_mint = (burned + self.silos * mask[..., np.newaxis]) * self.portion_in_uni[..., np.newaxis]
        used = self.position.mint(to_mint[..., 0], to_mint[..., 1])
        self.silos += burned - used

        # print((amounts[...,1] / (amounts[...,1] + amounts[...,0] * price)).mean())

        return amounts
