import numpy as np
import random
import math


def gen_wires_tmp2(n_data, length, imsize):

    xx = np.zeros([n_data, imsize, imsize, 1], dtype=np.double)
    yy = np.zeros([n_data, int(imsize/2), int(imsize/2), 17], dtype=np.double)

    radius = 3

    for i in range(n_data * 2):
        if i % 50 == 0:

            x_loc = 1. * random.random() * (imsize - length) + length / 2.
            y_loc = 1. * random.random() * (imsize - length) + length / 2.

            angle = random.random() * np.pi

            vec_x = np.cos(angle)
            vec_y = np.sin(angle)

        for x in range(imsize):
            for y in range(imsize):
                dx = x - x_loc
                dy = y - y_loc

                parallel = vec_x * dx + vec_y * dy
                ortho = math.sqrt((dx - parallel * vec_x) ** 2 + (dy - parallel * vec_y) ** 2)
                parallel = math.fabs(parallel)

                if parallel <= length / 2. and ortho <= radius / 2.:
                    xx[i % n_data, x, y, 0] = 1
                    if x % 2 == 0 and y % 2 == 0:
                        yy[i % n_data, int(x / 2), int(y / 2), 0] = vec_x
                        yy[i % n_data, int(x / 2), int(y / 2), 5] = vec_x
                        yy[i % n_data, int(x / 2), int(y / 2), 1] = vec_y
                        yy[i % n_data, int(x / 2), int(y / 2), 4] = -vec_y
                        yy[i % n_data, int(x / 2), int(y / 2), 15] = 1
                        yy[i % n_data, int(x / 2), int(y / 2), 16] = 1

    xx += np.random.randn(n_data, imsize, imsize, 1) * 0.2
    yy += np.random.randn(n_data, int(imsize / 2), int(imsize / 2), 17) * 0.0001