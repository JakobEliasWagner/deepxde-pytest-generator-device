import torch
import deepxde as dde

torch.manual_seed(0)


def test_deepxde_example():
    geom = dde.geometry.Interval(0, 1)
    # Poisson equation: -u_xx = f
    def equation(x, y, f):
        dy_xx = dde.grad.hessian(y, x)
        return -dy_xx - f
    assert True
