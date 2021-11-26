from random import random

import numpy as np

from uniswap_simulator import Position


def coerce_to_tick_spacing(spacing, ticks):
    ticks = ticks.copy()
    ticks[...,0] -= np.mod(ticks[...,0], spacing)
    ticks[...,1] -= np.mod(ticks[...,1], spacing) - spacing
    return ticks


DEADBAND_U = 100
DEADBAND_L = 100


def get_hypothesis(price, touchpoint_lower, touchpoint_upper):
    touchpoint_lower_tick = np.log(touchpoint_lower) / np.log(1.0001)
    touchpoint_upper_tick = np.log(touchpoint_upper) / np.log(1.0001)
    touchpoint_center_tick = (touchpoint_lower_tick + touchpoint_upper_tick) / 2
    touchpoint_width = touchpoint_upper_tick - touchpoint_lower_tick

    hypothesis_lower_tick = np.zeros_like(touchpoint_lower_tick)
    hypothesis_upper_tick = np.zeros_like(touchpoint_upper_tick)

    mask = price > touchpoint_center_tick
    hypothesis_lower_tick[mask] = touchpoint_upper_tick[mask] + DEADBAND_U
    hypothesis_lower_tick[~mask] = touchpoint_lower_tick[~mask] - DEADBAND_L - touchpoint_width[~mask]
    hypothesis_upper_tick = touchpoint_lower_tick + touchpoint_width

    return (
        np.power(1.0001, hypothesis_lower_tick),
        np.power(1.0001, hypothesis_upper_tick)
    )




class DeadbandStrategy:
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

        self.silo0 = np.zeros_like(price)
        self.silo1 = np.zeros_like(price)

    def reset(self, price):
        self.position.reset(price)
        self.limit_order = Position(price, self.position.lower, price / 1.0001, self.position.fee)

        self.silo0 = np.zeros_like(price)
        self.silo1 = np.zeros_like(price)

    def mint(self, amount0, amount1):
        fraction = 1 - (1.0001 ** (-self.half_width / 2))
        used0, used1 = self.position.mint(amount0 * fraction, amount1 * fraction)
        
        self.silo0 += amount0 - used0
        self.silo1 += amount1 - used1

        return amount0, amount1

    def update(self, price):
        amounts = self.position.update(price)
        amounts += self.limit_order.update(price)
        amounts[...,0] += self.silo0
        amounts[...,1] += self.silo1

        return amounts
