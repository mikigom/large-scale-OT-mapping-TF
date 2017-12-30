import numpy as np
import random


class GeneratorGaussians4(object):
    def __init__(self,
                 batch_size: int=256,
                 scale: float=2.,
                 center_coor_min: float=-0.25,
                 center_coor_max: float=+0.25,
                 stdev: float=1.414):
        self.batch_size = batch_size
        self.stdev = stdev
        scale = scale
        diag_len = np.sqrt(center_coor_min**2 + center_coor_max**2)
        centers = [
            (center_coor_max / diag_len, center_coor_max / diag_len),
            (center_coor_max / diag_len, center_coor_min / diag_len),
            (center_coor_min / diag_len, center_coor_max / diag_len),
            (center_coor_min / diag_len, center_coor_min / diag_len)
        ]
        self.centers = [(scale * x, scale * y) for x, y in centers]

    def __iter__(self):
        while True:
            dataset = []
            for i in range(self.batch_size):
                point = np.random.randn(2) * .02
                center = random.choice(self.centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= self.stdev
            yield dataset


class GeneratorGaussian1(object):
    def __init__(self,
                 batch_size=256):
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield np.random.multivariate_normal((0, 0), ((0.15, 0), (0, 0.15)), self.batch_size)
