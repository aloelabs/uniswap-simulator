from random import random

import numpy as np

from uniswap_simulator import Position


def coerce_to_tick_spacing(spacing, ticks):
    ticks = ticks.copy()
    ticks[...,0] -= np.mod(ticks[...,0], spacing)
    ticks[...,1] -= np.mod(ticks[...,1], spacing) - spacing
    return ticks


class DRDP0Strategy:
    limit_order_width = 10
    epsilon = 0.001

    def __init__(self, price, lower, upper, fee):
        self._tick_spacing = 10
        if fee == 0.3 / 100:
            self._tick_spacing = 60
        elif fee == 1.0 / 100:
            self._tick_spacing = 100

        self.half_width = (np.log(upper) - np.log(lower)) / (2 * np.log(1.0001))
        self.position = Position(price, lower, upper, fee)
        self.limit_order = Position(price, lower, price / 1.0001, fee)

    def reset(self, price):
        self.position.reset(price)
        self.limit_order = Position(price, self.position.lower, price / 1.0001, self.position.fee)

    def mint(self, amount0, amount1):
        return self.position.mint(amount0, amount1)

    def update(self, price):
        amounts = self.position.update(price)
        amounts += self.limit_order.update(price)

        self._compound(price, amounts.copy())

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

    def _compound(self, price, amounts, fraction=0.99):
        inactive_limit_orders = np.any((
            price < self.limit_order.lower,
            price > self.limit_order.upper
        ), axis=0)

        earned = self.position._earned.copy()
        earned += self.limit_order.burn()

        # change basis of `amounts[...,0]` so that it represents wealth in each asset
        amounts[...,0] *= price

        if random() < 0.01:
            print((amounts[...,0] / amounts.sum(axis=-1)).mean())
        excess0 = amounts[...,0] > amounts[...,1]
        # compute active trading range (defined by lower and upper ticks)
        active_ticks = np.log(price) / np.log(1.0001)
        active_ticks = coerce_to_tick_spacing(
            self._tick_spacing,
            np.vstack((active_ticks, active_ticks)).T
        )

        w = max(DRDP0Strategy.limit_order_width, self._tick_spacing)
        new_bounds = np.zeros_like(active_ticks)
        new_bounds[excess0, 0] = active_ticks[excess0, 1]
        new_bounds[excess0, 1] = active_ticks[excess0, 1] + w
        new_bounds[~excess0, 0] = active_ticks[~excess0, 0] - w
        new_bounds[~excess0, 1] = active_ticks[~excess0, 0]
        new_bounds = np.power(1.0001, new_bounds)
        new_bounds = np.sqrt(new_bounds)
        
        m = np.sqrt(new_bounds[...,0] * new_bounds[...,1])
        x = np.zeros_like(m)
        y = np.zeros_like(m)
        
        # x will be equal to the target spend amount where excess0 is True.
        # it will be 0 everywhere else.
        # this doesn't hold for active limit orders, in which case x is maximized
        x[excess0] = (amounts[excess0, 0] - amounts[excess0, 1]) / \
            (price[excess0] + m[excess0])
        x = np.clip(x, a_min=0, a_max=earned[...,0] * fraction)
        x[~inactive_limit_orders] = earned[~inactive_limit_orders, 0] * fraction
        # y will be equal to the target spend amount where excess0 is False.
        # it will be 0 everywhere else.
        # this doesn't hold for active limit orders, in which case y is maximized
        y[~excess0] = (amounts[~excess0, 1] - amounts[~excess0, 0]) / \
            (price[~excess0] + m[~excess0])
        y[~excess0] *= m[~excess0]
        y = np.clip(y, a_min=0, a_max=earned[...,1] * fraction)
        y[~inactive_limit_orders] = earned[~inactive_limit_orders, 1] * fraction

        self.limit_order._lower_sqrt[inactive_limit_orders] = new_bounds[inactive_limit_orders, 0]
        self.limit_order._upper_sqrt[inactive_limit_orders] = new_bounds[inactive_limit_orders, 1]
        used = self.limit_order.mint(x, y)

        self.position._earned = earned - used
        assert self.position._earned.min(
        ) >= -DRDP0Strategy.epsilon, self.position._earned.min()
        self.position._earned[self.position._earned < 0] = 0
