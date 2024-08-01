# tests/test_focal_loss.py
import unittest
import numpy as np
from gbdtLoss.binary import FocalLoss
from gbdtLoss.utils import check_gradient

class TestFocalLossFunction(unittest.TestCase):
    def setUp(self):
        self.loss_function = FocalLoss(gamma=None, alpha=None)
        self.epsilon = 1e-5
        self.y_true = np.random.uniform(0, 1, 100) > .5
        self.y_pred = np.random.uniform(0, 1, 100)

    def test_first_order_gradient(self):
        for gamma in [0, 1, 2, 3]:
            for alpha in [.1, .3, .5, .7, .9]:
                self.loss_function.gamma = gamma
                self.loss_function.alpha = alpha
                diff = check_gradient(
                    func=lambda x: self.loss_function(self.y_true, x),
                    grad=lambda x: self.loss_function.grad(self.y_true, x) / (x * (1 - x)),
                    values=self.y_pred
                )
                self.assertAlmostEqual(diff, 0, delta=1e-5, msg=f"Fail with alpha: {alpha}, gamma: {gamma}")

    def test_second_order_gradient(self):
        for gamma in [0, 1, 2, 3]:
            for alpha in [.1, .3, .5, .7, .9]:
                self.loss_function.gamma = gamma
                self.loss_function.alpha = alpha
                diff = check_gradient(
                    func=lambda x: self.loss_function.grad(self.y_true, x),
                    grad=lambda x: self.loss_function.hess(self.y_true, x) / (x * (1 - x)),
                    values=self.y_pred
                )
                self.assertAlmostEqual(diff, 0, delta=1e-5, msg=f"Fail with alpha: {alpha}, gamma: {gamma}")

if __name__ == '__main__':
    unittest.main()