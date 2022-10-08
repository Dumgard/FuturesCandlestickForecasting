import matplotlib.pyplot as plt
import numpy as np


def plot_group(y_group: list[list[float]],
               x: list[float] = None,
               ):
    y_group = np.array(y_group)
    if len(y_group.shape) < 2:
        y_group = y_group.reshape(1, -1)

    if x is None:
        x = np.linspace(0, 1000, y_group.shape[-1])

    # num_x == num_y
    assert len(x) == y_group.shape[-1]

    fig = plt.figure()

    x = np.array(x).reshape(-1)
    for i in range(y_group.shape[0]):
        plt.plot(x, y_group[i], label=f'epoch {i + 1}')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    a1 = [[3, 6, 8, 10, 12], [9, 8, 7, 6, 5], [3, 4, 3, 4, 10]]
    a2 = [1, 5, 3, 4, 2]
    plot_group(a1)
    plot_group(a2)
