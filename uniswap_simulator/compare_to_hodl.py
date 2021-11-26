import numpy as np


INITIAL_INVENTORY0 = 10000


def compare_to_hodl(strategy, prices, T):
    # price trajectories should start from the same value (at t=0)
    assert prices[0].std() == 0.
    initial_price = prices[0].mean()

    # array to store results
    amounts = np.zeros((*prices.shape, 2))

    # mint liquidity to get things rolling
    m0 = np.full_like(prices[0], INITIAL_INVENTORY0)
    m1 = np.full_like(prices[0], INITIAL_INVENTORY0 * initial_price)
    strategy.reset(prices[0])
    strategy.mint(m0, m1)

    # iterate through t=0 --> t=t_max
    for i in range(len(prices)):
        amounts[i] = strategy.update(prices[i])

    # 0th axis is time
    # 1st axis is trajectories
    # 2nd axis is [amount0, amount1]

    values = amounts.copy()
    values[..., 0] *= prices

    hodl = m0 * prices + m1
    y = values.copy()

    y = y.sum(axis=2) / hodl[0]
    G_end_point = np.log(y[-1]).mean() / T
    y = hodl[-1] / hodl[0]
    G_end_point_hodl = np.log(y).mean() / T

    return G_end_point, G_end_point_hodl
