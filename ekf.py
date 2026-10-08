import numpy as np

from motion_model import (
    cv_state_transition,
    cv_process_noise
)


def ekf_predict(
    x,
    P,
    dt,
    sigma_ax=1.0,
    sigma_ay=1.0
):

    F = cv_state_transition(dt)

    Q = cv_process_noise(
        dt,
        sigma_ax=sigma_ax,
        sigma_ay=sigma_ay
    )

    x_pred = F @ x

    P_pred = (
        F @ P @ F.T
        + Q
    )

    P_pred = 0.5 * (
        P_pred + P_pred.T
    )

    return x_pred, P_pred