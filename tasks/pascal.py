import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.zeros(360, dtype=np.float32)
    x[180] = 1.0

    T0 = np.identity(360, dtype=np.float32)
    T = T0.copy()
    n = 5
    for i in range(1, n + 1):
        T = T + np.concatenate([T0[:, i:], T0[:, 0:i]], axis=1)
        T = T + np.concatenate([T0[:, -i:], T0[:, 0:-i]], axis=1)
    T = T / (2 * n + 1)
    print(T[180, ...])
    for i in range(100):
        x = np.matmul(T, x)

    plt.plot(x, '-r')
    plt.show()
