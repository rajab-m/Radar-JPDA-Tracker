import numpy as np


def wrap_to_pi(angle):

    return (
        (angle + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )


def normalize_angle_residual(nu):

    nu = nu.copy()

    nu[1] = wrap_to_pi(nu[1])

    return nu


def radar_measurement_model(x):

    px, py, vx, vy = x

    r_squared = px**2 + py**2

    r = np.sqrt(
        max(r_squared, 1e-12)
    )

    azimuth = np.arctan2(
        py,
        px
    )

    radial_velocity = (
        px * vx + py * vy
    ) / r

    return np.array(
        [
            r,
            azimuth,
            radial_velocity
        ],
        dtype=float
    )


def radar_measurement_jacobian(x):

    px, py, vx, vy = x

    c1 = (
        px**2
        + py**2
    )

    if c1 < 1e-12:

        raise ValueError(
            "State is too close to radar origin; "
            "radar Jacobian is undefined."
        )

    r = np.sqrt(c1)

    r3 = c1 * r

    H = np.array([
        [
            px / r,
            py / r,
            0.0,
            0.0
        ],

        [
            -py / c1,
            px / c1,
            0.0,
            0.0
        ],

        [
            py * (
                py * vx
                - px * vy
            ) / r3,

            px * (
                px * vy
                - py * vx
            ) / r3,

            px / r,
            py / r
        ]
    ], dtype=float)

    return H