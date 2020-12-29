from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf


def main():
    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return (
            dy_t
            - dy_xx
            + tf.exp(-x[:, 1:])
            * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
        )

    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
    ic = dde.IC(geomtime, func, lambda _, on_initial: on_initial)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=40,
        num_boundary=20,
        num_initial=10,
        train_distribution="pseudo",
        solution=func,
        num_test=10000,
    )

    layer_size = [2] + [32] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)

    for _ in range(5):
        model.compile("adam", lr=0.001, metrics=["l2 relative error"])
        model.train(epochs=2000)
        print("epoch = {}, resample train points...".format(model.train_state.epoch))
        data.resample_train_points()
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=2000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
