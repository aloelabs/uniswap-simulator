from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from uniswap_simulator import GeometricBrownianMotion, Position, compare_to_hodl

from .compounding_strategy import CompoundingStrategy


def get_performance(args):
    p0, mu, sigma, dt, T = args

    gbm = GeometricBrownianMotion(p0, mu, sigma, dt, T)
    prices = gbm.sample(1000).astype('float64')
    time = np.arange(start=0, stop=prices.shape[0])

    strategy = Position(prices[0], 1.0001 ** -887272,
                        1.0001 ** 887272, 1.0/100)
    # strategy = CompoundingStrategy(prices[0], 1.0001 ** -887272,
    #                     1.0001 ** 887272, 0.05/100)

    assert np.all(prices < 1.0001 ** 887272), prices.max()
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
    ax.view_init(32, -121)

    # Plot the surface
    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid[..., 0] - z_grid[..., 1],
        cmap='rainbow',
        linewidth=0,
        antialiased=True
    )
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\sigma$')
    ax.set_zlabel('$G - G_{hodl}$')
    ax.set_zlim3d(-.01, +.01)

    plt.savefig('results/surf.png')


if __name__ == '__main__':
    main()
