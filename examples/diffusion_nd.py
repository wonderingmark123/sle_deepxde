from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf
tf.compat.v1.disable_eager_execution()

def main():
    
    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x,i=0, j=4)
        dy_xx = dde.grad.hessian(y, x,component=0 , j=0)
        # dy_xx = dde.grad.hessian(y, x , j=0)
        return (
            dy_t
            - dy_xx
            + tf.exp(-x[:, 4:])
            * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1])),
            x[:, 0:1] * 0,
        )

    def func(x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 4:])
    def func2(x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 4:]),0

    # geom = dde.geometry.Interval(-1, 1)
    geom = dde.geometry.Rectangle
    geom = dde.geometry.Hypercube([-1]*4,[1]*4)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary,component=0)
    ic = dde.IC(geomtime, func, lambda _, on_initial: on_initial,component=0)
    ic2 = dde.IC(geomtime,lambda shit:  1, lambda _, on_initial: on_initial,component=1)
    # observe_x = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 1))).T
    # ptset = dde.bc.PointSet(observe_x)
    # observe_y = dde.DirichletBC(
    #     geomtime, ptset.values_to_func(func(observe_x)), lambda x, _: ptset.inside(x)
    # )

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic,ic2],
        num_domain=4000,
        num_boundary=1,
        num_initial=100,
    )
    
    layer_size = [5] + [32] * 3 + [2]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)

    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=10000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
