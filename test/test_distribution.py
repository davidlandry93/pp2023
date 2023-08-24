import math
import torch
import unittest

from pp2023.distribution import quantiles_to_deltas, deltas_to_quantiles


class AdditiveQuantileTest(unittest.TestCase):
    def test_quantiles_to_deltas(self):
        quantiles = torch.tensor([21, 23, 25, 28])
        deltas = quantiles_to_deltas(quantiles)

        target_value = torch.log1p(torch.tensor([2, 2, 1e-6, 3]))
        target_value[2] = 25

        self.assertTrue(torch.allclose(deltas, target_value))

    def test_deltas_to_quantiles(self):
        deltas = torch.tensor([math.log(2 + 1), 25, math.log(3 + 1)])
        quantiles = deltas_to_quantiles(deltas)

        target_values = torch.tensor([23.0, 25.0, 28.0])
        self.assertTrue(torch.allclose(quantiles, target_values))
