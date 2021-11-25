import numpy as np

from uniswap_simulator.liquidity_amounts import liquidity_for_amounts, amounts_for_liquidity


class Position:
    def __init__(self, price, lower, upper, fee):
        self._price_sqrt = np.sqrt(price)
        self._lower_sqrt = np.sqrt(lower)
        self._upper_sqrt = np.sqrt(upper)
        self._fee = fee

        self._liquidity = np.zeros_like(price)
        self._earned = None

    def reset(self, price):
        self._price_sqrt = np.sqrt(price)
        self._liquidity = np.zeros_like(price)
        self._earned = None

    @property
    def _price(self):
        return np.square(self._price_sqrt)

    @property
    def lower(self):
        return np.square(self._lower_sqrt)

    @property
    def upper(self):
        return np.square(self._upper_sqrt)

    @property
    def fee(self):
        return self._fee

    @property
    def amounts(self):
        return self._earned + amounts_for_liquidity(
            self._price_sqrt,
            self._lower_sqrt,
            self._upper_sqrt,
            self._liquidity
        )

    @property
    def collectable(self):
        return self._earned

    def update(self, price):
        # If price movement is less than fee, it's not guaranteed that the AMM will
        # be arb'd to match new price
        should_update = np.absolute(price / self._price - 1) >= self._fee
        price[~should_update] = self._price[~should_update]
        price_sqrt = np.sqrt(price)

        amounts_previous = amounts_for_liquidity(
            self._price_sqrt.clip(min=self._lower_sqrt, max=self._upper_sqrt),
            self._lower_sqrt,
            self._upper_sqrt,
            self._liquidity
        )
        amounts_current = amounts_for_liquidity(
            price_sqrt.clip(min=self._lower_sqrt, max=self._upper_sqrt),
            self._lower_sqrt,
            self._upper_sqrt,
            self._liquidity
        )

        diff = amounts_current - amounts_previous
        mask = diff[..., 0] > 0
        if self._earned is None:
            self._earned = np.zeros_like(diff)
        
        self._earned[mask, 0] += diff[mask, 0] * self._fee
        self._earned[~mask, 1] += diff[~mask, 1] * self._fee

        self._price_sqrt = price_sqrt
        return self.amounts

    def mint(self, amount0, amount1):
        liquidity = liquidity_for_amounts(
            self._price_sqrt,
            self._lower_sqrt,
            self._upper_sqrt,
            amount0,
            amount1
        )
        self._liquidity += liquidity
        return amounts_for_liquidity(
            self._price_sqrt,
            self._lower_sqrt,
            self._upper_sqrt,
            liquidity
        )

    def burn(self, fraction=1.0):
        liquidity_to_burn = self._liquidity * fraction
        self._liquidity -= liquidity_to_burn

        burned = amounts_for_liquidity(
            self._price_sqrt,
            self._lower_sqrt,
            self._upper_sqrt,
            liquidity_to_burn
        )
        earned = self._earned * fraction

        self._earned -= earned
        return burned + earned

    def burn_at(self, mask, fraction=1.0):
        liquidity_to_burn = self._liquidity * fraction * mask
        self._liquidity -= liquidity_to_burn

        burned = amounts_for_liquidity(
            self._price_sqrt,
            self._lower_sqrt,
            self._upper_sqrt,
            liquidity_to_burn
        )
        earned = self._earned * fraction * mask[..., np.newaxis]

        self._earned -= earned
        return burned + earned
