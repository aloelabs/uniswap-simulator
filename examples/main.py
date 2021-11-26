from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from uniswap_simulator import GeometricBrownianMotion, Position, compare_to_hodl

from .strategies.static_main_position.compounding_strategy import CompoundingStrategy


MIN_TICK = -887272
MAX_TICK = +887272


def get_performance(args):
    p0, mu, sigma, dt, T = args

    gbm = GeometricBrownianMotion(p0, mu, sigma, dt, T)
    prices = gbm.sample(1000).astype('float64')
    prices = np.clip(
        prices,
        a_min=1.0001 ** MIN_TICK,
        a_max=1.0001 ** MAX_TICK
    )
    time = np.arange(start=0, stop=prices.shape[0])

    lower = np.full_like(prices[0], 1.0001 ** (MIN_TICK / 256))
    upper = np.full_like(prices[0], 1.0001 ** (MAX_TICK / 256))

    # strategy = Position(prices[0], lower, upper, 1.00/100)
    strategy = CompoundingStrategy(prices[0], lower, upper, 0.01/100)

    return np.array(compare_to_hodl(strategy, prices, time))


def main():
    # Geometric Brownian Motion parameters
    p0 = 1
    sigmas = np.linspace(0.1, 2.0, 20)
    mus = np.linspace(-0.8, 2.0, 20)

    x_grid, y_grid = np.meshgrid(mus, sigmas)
    z_grid = np.zeros((*x_grid.shape, 2))

    dt = 1. / 60.
    T = 31.

    ij = []
    args = []
    for i in range(len(sigmas)):
        for j in range(len(mus)):
            ij.append((i, j))
            args.append((p0, mus[j], sigmas[i], dt, T))

    with Pool(12) as p:
        performances = p.map(get_performance, args)
        for k, performance in enumerate(performances):
            z_grid[ij[k][0], ij[k][1]] = performance

    # Save simulation results
    np.save('results/xgrid.npy', x_grid)
    np.save('results/ygrid.npy', y_grid)
    np.save('results/zgrid.npy', z_grid)
    print(z_grid.max())

    plt.clf()
    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.view_init(32, -130)

    # Plot the surface
    Z = z_grid[..., 0] - z_grid[..., 1]
    ax.plot_surface(
        x_grid,
        y_grid,
        Z,
        cmap='rainbow',
        linewidth=0,
        antialiased=True
    )
    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array(Z)
    plt.colorbar(m)

    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\sigma$')
    ax.set_zlabel('$G - G_{HODL}$')
    ax.set_zlim3d(-.01, +.01)

    plt.savefig('results/surf.png')


if __name__ == '__main__':
    main()
