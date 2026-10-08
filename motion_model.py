import numpy as np


def cv_state_transition(dt):

    return np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)


def cv_process_noise(
    dt,
    sigma_ax=1.0,
    sigma_ay=1.0
):

    dt2 = dt**2
    dt3 = dt**3
    dt4 = dt**4

    qx = sigma_ax**2
    qy = sigma_ay**2

    return np.array([
        [
            qx * dt4 / 4.0,
            0.0,
            qx * dt3 / 2.0,
            0.0
        ],

        [
            0.0,
            qy * dt4 / 4.0,
            0.0,
            qy * dt3 / 2.0
        ],

        [
            qx * dt3 / 2.0,
            0.0,
            qx * dt2,
            0.0
        ],

        [
            0.0,
            qy * dt3 / 2.0,
            0.0,
            qy * dt2
        ]
    ], dtype=float)