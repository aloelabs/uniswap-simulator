import numpy as np


def liquidity_for_amount0(sqrt_ratio_a, sqrt_ratio_b, amount0):
    """
    Computes the amount of liquidity received for a given amount of token0 and price range.
    Array sizes of all input args are assumed to match.
    """
    mul = sqrt_ratio_a * sqrt_ratio_b
    sub = sqrt_ratio_b - sqrt_ratio_a

    mask = sqrt_ratio_a < sqrt_ratio_b
    liquidity = np.full_like(amount0, np.nan)
    liquidity[mask] = (amount0 * mul / sub)[mask]
    return liquidity


def liquidity_for_amount1(sqrt_ratio_a, sqrt_ratio_b, amount1):
    """
    Computes the amount of liquidity received for a given amount of token1 and price range
    Array sizes of all input args are assumed to match.
    """
    mask = sqrt_ratio_a < sqrt_ratio_b
    liquidity = np.full_like(amount1, np.nan)
    liquidity[mask] = (amount1 / (sqrt_ratio_b - sqrt_ratio_a))[mask]
    return liquidity


def liquidity_for_amounts(sqrt_ratio, sqrt_ratio_a, sqrt_ratio_b, amount0, amount1):
    """
    Computes the maximum amount of liquidity received for a given amount of token0, token1, the current
    pool prices and the prices at the tick boundaries.
    Array sizes of all input args are assumed to match.
    """
    liquidity = np.zeros_like(sqrt_ratio)

    mask = sqrt_ratio < sqrt_ratio_b
    liquidity[mask] = np.minimum(
        liquidity_for_amount0(sqrt_ratio, sqrt_ratio_b, amount0),
        liquidity_for_amount1(sqrt_ratio_a, sqrt_ratio, amount1)
    )[mask]

    mask = sqrt_ratio <= sqrt_ratio_a
    liquidity[mask] = liquidity_for_amount0(sqrt_ratio_a, sqrt_ratio_b, amount0)[mask]

    mask = sqrt_ratio >= sqrt_ratio_b
    liquidity[mask] = liquidity_for_amount1(sqrt_ratio_a, sqrt_ratio_b, amount1)[mask]

    assert not np.any(np.isnan(liquidity))
    return liquidity


def amount0_for_liquidity(sqrt_ratio_a, sqrt_ratio_b, liquidity):
    """
    Computes the amount of token0 for a given amount of liquidity and a price range.
    Array sizes of all input args are assumed to match.
    """
    mul = sqrt_ratio_a * sqrt_ratio_b
    sub = sqrt_ratio_b - sqrt_ratio_a

    mask = sqrt_ratio_a < sqrt_ratio_b
    amount0 = np.full_like(liquidity, np.nan)
    amount0[mask] = (liquidity * sub / mul)[mask]
    return amount0


def amount1_for_liquidity(sqrt_ratio_a, sqrt_ratio_b, liquidity):
    """
    Computes the amount of token1 for a given amount of liquidity and a price range.
    Array sizes of all input args are assumed to match.
    """
    mask = sqrt_ratio_a < sqrt_ratio_b
    amount1 = np.full_like(liquidity, np.nan)
    amount1[mask] = (liquidity * (sqrt_ratio_b - sqrt_ratio_a))[mask]
    return amount1


def amounts_for_liquidity(sqrt_ratio, sqrt_ratio_a, sqrt_ratio_b, liquidity):
    """
    Computes the token0 and token1 value for a given amount of liquidity, the current
    pool prices and the prices at the tick boundaries.
    Array sizes of all input args are assumed to match.
    """
    amounts = np.zeros((*liquidity.shape, 2))

    mask = sqrt_ratio < sqrt_ratio_b
    amounts[mask, 0] = amount0_for_liquidity(sqrt_ratio, sqrt_ratio_b, liquidity)[mask]
    amounts[mask, 1] = amount1_for_liquidity(sqrt_ratio_a, sqrt_ratio, liquidity)[mask]

    mask = sqrt_ratio <= sqrt_ratio_a
    amounts[mask, 0] = amount0_for_liquidity(sqrt_ratio_a, sqrt_ratio_b, liquidity)[mask]
    amounts[mask, 1] = 0

    mask = sqrt_ratio >= sqrt_ratio_b
    amounts[mask, 0] = 0
    amounts[mask, 1] = amount1_for_liquidity(sqrt_ratio_a, sqrt_ratio_b, liquidity)[mask]

    assert not np.any(np.isnan(amounts))
    return amounts
