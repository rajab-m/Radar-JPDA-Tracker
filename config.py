from dataclasses import dataclass
import numpy as np


@dataclass
class TrackerConfig:

    # ========================================================
    # RADAR
    # ========================================================

    sigma_range: float = 0.3
    sigma_angle: float = np.deg2rad(1.0)
    sigma_doppler: float = 0.17

    # ========================================================
    # MOTION MODEL
    # ========================================================

    sigma_ax: float = 1.0
    sigma_ay: float = 1.0

    # ========================================================
    # JPDA
    # ========================================================

    pd: float = 0.90
    pg: float = 0.99
    clutter_density: float = 1e-4

    # ========================================================
    # ASSOCIATION
    # ========================================================

    association_probability_threshold: float = 0.70

    # ========================================================
    # TRACK MANAGEMENT
    # ========================================================

    max_missed: int = 6
    confirmation_hits: int = 4
    min_range: float = 0.2
    max_tracks: int = 40
    # if the measurement has an association_prob with any of the existing tracks > measurement_probability no new tracks will be created
    # 0 means if the meas falls in a gate of any track associate it and don't create no track, in this case a new track will be opened only and only
    # if the gate is false for all tracks, increasing this parameter will make the association more strict
    measurement_probability: float = 0.0       

    # ========================================================
    # MERGE
    #
    # ONLY POSITION EUCLIDEAN DISTANCE.
    # NO VELOCITY.
    # ========================================================

    merge_distance_threshold: float = 2.0
    merge_confirmed_only: bool = False

    # ========================================================
    # TIMING
    # ========================================================

    default_dt: float = 0.1

    # ========================================================
    # INITIALIZATION
    # ========================================================

    minimum_initial_velocity_sigma: float = 5.0
    initial_velocity_sigma_multiplier: float = 3.0

    # ========================================================
    # NUMERICAL
    # ========================================================

    covariance_eigenvalue_floor: float = 1e-10

    # ========================================================
    # DEBUG
    # ========================================================

    debug: bool = True

    @property
    def R(self):

        return np.diag([
            self.sigma_range ** 2,
            self.sigma_angle ** 2,
            self.sigma_doppler ** 2
        ])