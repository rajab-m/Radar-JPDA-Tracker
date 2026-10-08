import numpy as np

from radar_model import (
    radar_measurement_model,
    radar_measurement_jacobian,
    normalize_angle_residual,
)

from ekf import ekf_predict

from jpda import JPDA

from track_manager import TrackManager


class RadarJPDAEKF:

    def __init__(self, config):

        self.config = config

        self.meas_prob = (
            config.measurement_probability
        )

        # ====================================================
        # TRACK MANAGER
        # ====================================================

        self.track_manager = TrackManager(
            R=config.R,

            max_tracks=config.max_tracks,

            confirmation_hits=(
                config.confirmation_hits
            ),

            max_missed=config.max_missed,

            min_range=config.min_range,

            merge_distance_threshold=(
                config.merge_distance_threshold
            ),
            debug = config.debug,

            merge_confirmed_only=(
                config.merge_confirmed_only
            ),

            minimum_initial_velocity_sigma=(
                config.minimum_initial_velocity_sigma
            ),

            initial_velocity_sigma_multiplier=(
                config.initial_velocity_sigma_multiplier
            )
        )

        # ====================================================
        # JPDA
        # ====================================================

        self.jpda = JPDA(
            pd=config.pd,
            pg=config.pg,
            clutter_density=config.clutter_density,
            measurement_dimension=3
        )

        self.previous_timestamp = None

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_tracks(self, dt):

        active_ids = (
            self.track_manager
            .get_active_track_indices()
        )

        for track_id in active_ids:

            track = (
                self.track_manager
                .tracks[track_id]
            )

            track.x, track.P = ekf_predict(
                track.x,
                track.P,
                dt,
                sigma_ax=self.config.sigma_ax,
                sigma_ay=self.config.sigma_ay
            )

    # ========================================================
    # ASSOCIATION QUANTITIES
    # ========================================================

    def calculate_association_quantities(
        self,
        measurements
    ):

        active_ids = (
            self.track_manager
            .get_active_track_indices()
        )

        n_tracks = len(active_ids)

        n_measurements = len(measurements)

        mahalanobis_matrix = np.full(
            (
                n_tracks,
                n_measurements
            ),
            np.inf
        )

        likelihood_matrix = np.zeros(
            (
                n_tracks,
                n_measurements
            )
        )

        innovations = [
            [
                None
                for _ in range(n_measurements)
            ]
            for _ in range(n_tracks)
        ]

        innovation_covariances = [
            [
                None
                for _ in range(n_measurements)
            ]
            for _ in range(n_tracks)
        ]

        gains = [
            [
                None
                for _ in range(n_measurements)
            ]
            for _ in range(n_tracks)
        ]

        jacobians = [
            [
                None
                for _ in range(n_measurements)
            ]
            for _ in range(n_tracks)
        ]

        predicted_measurements = [
            [
                None
                for _ in range(n_measurements)
            ]
            for _ in range(n_tracks)
        ]

        # ====================================================
        # CALCULATE TRACK / MEASUREMENT QUANTITIES
        # ====================================================

        for i, track_id in enumerate(active_ids):

            track = (
                self.track_manager
                .tracks[track_id]
            )

            z_pred = radar_measurement_model(
                track.x
            )

            H = radar_measurement_jacobian(
                track.x
            )

            for j, measurement_data in enumerate(
                measurements
            ):

                measurement = (
                    measurement_data["z"]
                )

                R_measurement = (
                    measurement_data["R"]
                )

                S = (
                    H @ track.P @ H.T
                    + R_measurement
                )

                S = 0.5 * (
                    S + S.T
                )

                PHt = (
                    track.P @ H.T
                )

                K = np.linalg.solve(
                    S,
                    PHt.T
                ).T

                innovation = (
                    measurement
                    - z_pred
                )

                innovation = (
                    normalize_angle_residual(
                        innovation
                    )
                )

                solved_nu = np.linalg.solve(
                    S,
                    innovation
                )

                D2 = float(
                    innovation.T
                    @ solved_nu
                )

                sign, logdet = (
                    np.linalg.slogdet(S)
                )

                if sign <= 0:

                    raise np.linalg.LinAlgError(
                        "Innovation covariance "
                        "is not positive definite."
                    )

                log_likelihood = (
                    -0.5 * D2
                    -0.5 * logdet
                    -0.5 * 3
                    * np.log(2.0 * np.pi)
                )

                likelihood = np.exp(
                    log_likelihood
                )

                mahalanobis_matrix[
                    i, j
                ] = D2

                likelihood_matrix[
                    i, j
                ] = likelihood

                innovations[
                    i
                ][j] = innovation

                innovation_covariances[
                    i
                ][j] = S

                gains[
                    i
                ][j] = K

                jacobians[
                    i
                ][j] = H

                predicted_measurements[
                    i
                ][j] = z_pred

        return (
            active_ids,
            mahalanobis_matrix,
            likelihood_matrix,
            innovations,
            innovation_covariances,
            gains,
            jacobians,
            predicted_measurements
        )

    # ========================================================
    # EKF / JPDA UPDATE
    # ========================================================

    def update_one_track(
        self,
        track,
        beta,
        innovations,
        innovation_covariances,
        gains
    ):

        beta_missed = float(
            beta[0]
        )

        beta_measurements = (
            beta[1:]
        )

        nu_bar = np.zeros_like(
            track.x,
            dtype=float
        )

        # ====================================================
        # MEAN STATE INNOVATION
        # ====================================================

        for j, beta_ij in enumerate(
            beta_measurements
        ):

            if beta_ij <= 0.0:
                continue

            nu = innovations[j]

            K = gains[j]

            nu_bar += (
                beta_ij
                * (K @ nu)
            )

        track.x = (
            track.x
            + nu_bar
        )

        # ====================================================
        # ASSOCIATION COVARIANCE
        # ====================================================

        state_innovation_second_moment = (
            np.zeros((4, 4))
        )

        for j, beta_ij in enumerate(
            beta_measurements
        ):

            if beta_ij <= 0.0:
                continue

            nu = innovations[j]

            K = gains[j]

            state_innovation = (
                K @ nu
            )

            state_innovation_second_moment += (
                beta_ij
                * np.outer(
                    state_innovation,
                    state_innovation
                )
            )

        association_covariance = (
            state_innovation_second_moment
            -
            np.outer(
                nu_bar,
                nu_bar
            )
        )

        association_covariance = (
            0.5
            * (
                association_covariance
                + association_covariance.T
            )
        )

        # ====================================================
        # MEASUREMENT COVARIANCE REDUCTION
        # ====================================================

        measurement_covariance_reduction = (
            np.zeros((4, 4))
        )

        for j, beta_ij in enumerate(
            beta_measurements
        ):

            if beta_ij <= 0.0:
                continue

            K = gains[j]

            S = innovation_covariances[j]

            measurement_covariance_reduction += (
                beta_ij
                * K
                @ S
                @ K.T
            )

        # ====================================================
        # NEW COVARIANCE
        # ====================================================

        P_new = (
            track.P
            - measurement_covariance_reduction
            + association_covariance
        )

        P_new = 0.5 * (
            P_new + P_new.T
        )

        # ====================================================
        # PSD FLOOR
        # ====================================================

        eigenvalues, eigenvectors = (
            np.linalg.eigh(P_new)
        )

        eigenvalues = np.maximum(
            eigenvalues,
            self.config.covariance_eigenvalue_floor
        )

        track.P = (
            eigenvectors
            @ np.diag(eigenvalues)
            @ eigenvectors.T
        )

        track.P = 0.5 * (
            track.P + track.P.T
        )

        # ====================================================
        # TRACK HIT / MISS MANAGEMENT
        # ====================================================

        association_probability = (
            1.0 - beta_missed
        )

        if (
            association_probability
            > self.config.association_probability_threshold
        ):

            track.missed = 0

            track.hits += 1

        else:

            track.missed += 1

        track.age += 1

        # ====================================================
        # CONFIRMATION
        # ====================================================

        if (
            track.hits
            >= self.config.confirmation_hits
        ):

            track.confirmed = True

    # ========================================================
    # VALIDATE MEASUREMENTS
    # ========================================================

    def validate_measurements(
        self,
        measurements
    ):

        if measurements is None:

            measurements = []

        valid_measurements = []

        for measurement_data in measurements:

            if not isinstance(
                measurement_data,
                dict
            ):

                raise ValueError(
                    "Each measurement must contain "
                    "{'z': ..., 'R': ...}."
                )

            measurement = np.asarray(
                measurement_data["z"],
                dtype=float
            )

            R_measurement = np.asarray(
                measurement_data["R"],
                dtype=float
            )

            if measurement.shape != (3,):

                raise ValueError(
                    "Each radar measurement must contain "
                    "[range, azimuth, radial_velocity]."
                )

            if R_measurement.shape != (3, 3):

                raise ValueError(
                    "Each measurement R must have shape "
                    "(3, 3)."
                )

            if not np.isfinite(
                measurement
            ).all():

                continue

            if not np.isfinite(
                R_measurement
            ).all():

                continue

            if (
                measurement[0]
                < self.config.min_range
            ):

                continue

            valid_measurements.append({
                "z": measurement,
                "R": R_measurement
            })

        return valid_measurements

    # ========================================================
    # CREATE NEW TRACKS
    # ========================================================

    def create_new_tracks(
        self,
        measurements,
        beta
    ):

        if len(measurements) == 0:
            return

        n_active = (
            self.track_manager
            .number_of_active_tracks()
        )

        for j in range(
            len(measurements)
        ):

            if n_active == 0:

                measurement_probability = 0.0

            else:

                measurement_probability = (
                    np.sum(
                        beta[:, j + 1]
                    )
                )

            # =================================================
            # KEEPING YOUR ORIGINAL BEHAVIOR
            # =================================================

            if (
                measurement_probability
                > self.meas_prob
            ):

                continue
            if self.config.debug:

                print(
                    "********************************"
                    "create New Track"
                    "********************************"
                )

                print(
                    measurements[j]["z"]
                )

            track = (
                self.track_manager
                .initialize_track(
                    measurements[j]["z"],
                    measurements[j]["R"]
                )
            )

            if track is not None:

                n_active += 1
                if self.config.debug:
                    print(
                        "New Track id is",
                        track.track_id
                    )

    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        measurements,
        timestamp
    ):

        # ====================================================
        # VALIDATE MEASUREMENTS
        # ====================================================

        measurements = (
            self.validate_measurements(
                measurements
            )
        )

        # ====================================================
        # DT
        # ====================================================

        if self.previous_timestamp is None:

            dt = self.config.default_dt

        else:

            dt = (
                timestamp
                - self.previous_timestamp
            )

            if dt <= 0.0:

                raise ValueError(
                    "Timestamps must increase."
                )

        self.previous_timestamp = timestamp

        # ====================================================
        # FIRST FRAME
        # ====================================================

        if (
            self.track_manager
            .number_of_active_tracks()
            == 0
        ):

            for measurement_data in measurements:

                self.track_manager.initialize_track(
                    measurement_data["z"],
                    measurement_data["R"]
                )

            self.track_manager.merge_close_tracks()

            return (
                self.track_manager
                .get_track_states()
            )

        # ====================================================
        # PREDICTION
        # ====================================================

        self.predict_tracks(dt)

        # ====================================================
        # NO MEASUREMENTS
        # ====================================================

        if len(measurements) == 0:

            active_ids = (
                self.track_manager
                .get_active_track_indices()
            )

            for track_id in active_ids:

                track = (
                    self.track_manager
                    .tracks[track_id]
                )

                track.missed += 1
                track.age += 1

            self.track_manager.delete_bad_tracks()

            return (
                self.track_manager
                .get_track_states()
            )

        # ====================================================
        # ACTIVE TRACKS
        # ====================================================

        active_ids = (
            self.track_manager
            .get_active_track_indices()
        )

        # ====================================================
        # DEBUGGING
        # ====================================================

        if self.config.debug:

            for i, track_id in enumerate(
                active_ids
            ):

                track = (
                    self.track_manager
                    .tracks[track_id]
                )

                z_pred = radar_measurement_model(
                    track.x
                )

                H = radar_measurement_jacobian(
                    track.x
                )

                print("\n==============================")
                print(f"TRACK {track_id}")
                print("x =", track.x)
                print("z_pred =", z_pred)

                for j, measurement_data in enumerate(
                    measurements
                ):

                    measurement = (
                        measurement_data["z"]
                    )

                    R_measurement = (
                        measurement_data["R"]
                    )

                    S = (
                        H @ track.P @ H.T
                        + R_measurement
                    )

                    S = 0.5 * (
                        S + S.T
                    )

                    innovation = (
                        measurement
                        - z_pred
                    )

                    innovation = (
                        normalize_angle_residual(
                            innovation
                        )
                    )

                    solved_nu = np.linalg.solve(
                        S,
                        innovation
                    )

                    D2 = float(
                        innovation.T
                        @ solved_nu
                    )

                    print(
                        f"\n  MEASUREMENT {j}"
                    )

                    print(
                        "  z =",
                        measurement
                    )

                    print("  R =")

                    print(
                        R_measurement
                    )

                    print(
                        "  innovation =",
                        innovation
                    )

                    print(
                        "  D2 =",
                        D2
                    )

        # ====================================================
        # ASSOCIATION QUANTITIES
        # ====================================================

        (
            active_ids,
            mahalanobis_matrix,
            likelihood_matrix,
            innovations,
            innovation_covariances,
            gains,
            jacobians,
            predicted_measurements
        ) = (
            self.calculate_association_quantities(
                measurements
            )
        )

        # ====================================================
        # JPDA
        # ====================================================

        (
            beta,
            gate_matrix,
            gamma,
            events,
            event_probabilities
        ) = self.jpda.associate(
            mahalanobis_matrix,
            likelihood_matrix
        )

        # ====================================================
        # JPDA DEBUG
        # ====================================================

        if self.config.debug:

            print(
                "GATE:\n",
                gate_matrix
            )

            print(
                "BETA:\n",
                beta
            )

        # ====================================================
        # UPDATE
        # ====================================================

        for i, track_id in enumerate(
            active_ids
        ):

            track = (
                self.track_manager
                .tracks[track_id]
            )

            self.update_one_track(
                track,
                beta[i],
                innovations[i],
                innovation_covariances[i],
                gains[i]
            )

        # ====================================================
        # NEW TRACKS
        # ====================================================

        self.create_new_tracks(
            measurements,
            beta
        )

        # ====================================================
        # MERGE
        #
        # TrackManager performs the actual merge.
        #
        # The merge criterion is Euclidean POSITION
        # distance only. Velocity is not used.
        # ====================================================

        self.track_manager.merge_close_tracks()

        # ====================================================
        # DELETE
        # ====================================================

        self.track_manager.delete_bad_tracks()

        # ====================================================
        # RETURN
        # ====================================================

        return (
            self.track_manager
            .get_track_states()
        )

    # ========================================================
    # PUBLIC HELPERS
    # ========================================================

    def get_track_states(self):

        return (
            self.track_manager
            .get_track_states()
        )

    def get_track_mask(self):

        return (
            self.track_manager
            .get_track_mask()
        )