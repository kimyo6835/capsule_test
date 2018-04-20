import numpy as np


def gen_wires(n_data, length, n_wires, imsize):
    x = np.zeros([n_data, imsize, imsize, 1], dtype=np.double)
    y = np.zeros([n_data, imsize, imsize, 17], dtype=np.double)

    for n in range(n_wires):
        x_loc = np.random.rand(n_data) * (imsize - length) + length / 2.
        y_loc = np.random.rand(n_data) * (imsize - length) + length / 2.
        angle = np.random.random(n_data) * np.pi
        vec_x = np.cos(angle)
        vec_y = np.sin(angle)

        x_wires = (vec_x.reshape(n_data, 1) * length / 2. * np.linspace(-1, 1, 100) + x_loc.reshape(n_data,
                                                                                                    1)).astype(int)
        y_wires = (vec_y.reshape(n_data, 1) * length / 2. * np.linspace(-1, 1, 100) + y_loc.reshape(n_data,
                                                                                                    1)).astype(int)

        indices_x = ((imsize * imsize * np.arange(n_data).reshape(n_data, 1)) + x_wires * imsize + y_wires).flatten()
        indices_y = ((imsize * imsize * np.shape(y)[-1] * np.arange(n_data).reshape(n_data,
                                                                                    1)) + x_wires * imsize *
                     np.shape(y)[-1] + y_wires * np.shape(y)[-1]).flatten()

        x = x.flatten()

        x[indices_x] = 1
        x = x.reshape(n_data, imsize, imsize, 1)

        y = y.flatten()

        y[indices_y] = vec_x.repeat(100).reshape(n_data, 100).flatten()
        y[indices_y + 1] = vec_y.repeat(100).reshape(n_data, 100).flatten()
        y[indices_y + 4] = -vec_y.repeat(100).reshape(n_data, 100).flatten()
        y[indices_y + 5] = vec_x.repeat(100).reshape(n_data, 100).flatten()

        y[indices_y + 15] = 1
        y[indices_y + 16] = 1

        y = y.reshape(n_data, imsize, imsize, 17)

        x += np.random.randn(n_data, imsize, imsize, 1) * 0.2
        y += np.random.randn(n_data, imsize, imsize, 17) * 0.001

    return x, y
