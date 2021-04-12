from __future__ import division
from functools import reduce
import numpy as np
import transformations as tf
from dq3d import quat, dualquat
from dq3d import op


def lbs(ws, tfs):
    return reduce(lambda x, p: x + p[0] * p[1],
                  zip(ws, tfs),
                  np.zeros((4, 4)))


if __name__ == "__main__":
    # 2 links simulation

    link_lengths = [1.0, 1.0]
    links = [dualquat(quat.identity()),
             dualquat(quat.identity(),
                      np.array([link_lengths[0], 0.0, 0.0]))]

    n_surface = 20
    surfaces1 = np.r_[[np.linspace(0.0, 2.0, n_surface),
                       0.2 * np.ones(n_surface),
                       np.zeros(n_surface)]].T
    surfaces2 = np.r_[[np.linspace(0.0, 2.0, n_surface),
                       -0.2 * np.ones(n_surface),
                       np.zeros(n_surface)]].T
    surfaces = np.concatenate([surfaces1, surfaces2], axis=0)
    print(surfaces)

    weight_coefficient = 2.0
    w_tmp = np.array([max(0.0, min(1.0, -weight_coefficient * (i / n_surface - 0.5) + 0.5)) for i in range(n_surface)])
    weights = np.tile(np.r_[[w_tmp, 1.0 - w_tmp]].T, (2, 1))
    print(weights)

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()


    def update_anim(i):
        plt.cla()
        theta = i * 0.5
        transform = dualquat(quat(*tf.quaternion_from_euler(0.0, 0.0, np.deg2rad(theta))))
        transformed_links_dual_quaternions = [links[0], links[1] * transform * links[1].inverse()]
        # for comparison, also compute matrix transformations
        transformed_links_matrices = [link.to_matrix() for link in transformed_links_dual_quaternions]
        # apply dual quaternions directly
        t_surfaces1 = np.array([op.dlb(w, transformed_links_dual_quaternions).transform_point(s) for w, s in zip(weights, surfaces)])
        # apply matrix transformations
        t_surfaces2 = np.array([np.dot(lbs(w, transformed_links_matrices), np.r_[s, 1.0])[:3] for w, s in zip(weights, surfaces)])

        plt.plot([0.0, link_lengths[0], link_lengths[1] * np.cos(np.deg2rad(theta)) + link_lengths[0]],
                 [0.0, 0.0, link_lengths[1] * np.sin(np.deg2rad(theta))], 'b-')
        plt.plot(t_surfaces2[:, 0], t_surfaces2[:, 1], 'go', label="LBS")
        plt.plot(t_surfaces1[:, 0], t_surfaces1[:, 1], 'ro', label="DLB")
        plt.xlim(-0.2, 2.2)
        plt.ylim(-0.7, 1.7)
        plt.legend()


    ani = animation.FuncAnimation(fig, update_anim, interval=10, frames=200)
    # ani.save('skinning.gif', writer="imagemagick")
    plt.show()
