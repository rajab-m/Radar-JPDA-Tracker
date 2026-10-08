import numpy as np

from models import Track


class TrackManager:

    def __init__(
        self,
        R,
        max_tracks=40,
        confirmation_hits=4,
        max_missed=6,
        min_range=0.2,
        merge_distance_threshold=2.0,
        merge_confirmed_only=False,
        debug = False,
        minimum_initial_velocity_sigma=5.0,
        initial_velocity_sigma_multiplier=3.0
    ):

        self.R = np.asarray(
            R,
            dtype=float
        )

        self.max_tracks = max_tracks

        self.confirmation_hits = (
            confirmation_hits
        )

        self.max_missed = max_missed

        self.min_range = min_range

        self.merge_distance_threshold = (
            merge_distance_threshold
        )

        self.merge_confirmed_only = (
            merge_confirmed_only
        )

        self.minimum_initial_velocity_sigma = (
            minimum_initial_velocity_sigma
        )

        self.initial_velocity_sigma_multiplier = (
            initial_velocity_sigma_multiplier
        )

        self.tracks_ID = [
            0 for _ in range(max_tracks)
        ]

        self.tracks = [
            None for _ in range(max_tracks)
        ]

    # ========================================================
    # ACTIVE TRACKS
    # ========================================================

    def get_active_track_indices(self):

        return [
            i
            for i, active in enumerate(
                self.tracks_ID
            )
            if active == 1
        ]

    def number_of_active_tracks(self):

        return int(
            np.sum(self.tracks_ID)
        )

    # ========================================================
    # FREE SLOT
    # ========================================================

    def get_free_track_id(self):

        for track_id in range(
            self.max_tracks
        ):

            if self.tracks_ID[track_id] == 0:

                return track_id

        return None

    # ========================================================
    # DISABLE
    # ========================================================

    def disable_track(
        self,
        track_id
    ):

        if (
            track_id < 0
            or track_id >= self.max_tracks
        ):

            return

        self.tracks_ID[track_id] = 0

        self.tracks[track_id] = None

    # ========================================================
    # INITIALIZE TRACK
    # ========================================================

    def initialize_track(
        self,
        measurement,
        R_measurement=None
    ):

        track_id = (
            self.get_free_track_id()
        )

        if track_id is None:
            return None

        r, azimuth, radial_velocity = (
            measurement
        )

        px = (
            r * np.cos(azimuth)
        )

        py = (
            r * np.sin(azimuth)
        )

        vx = (
            radial_velocity
            * np.cos(azimuth)
        )

        vy = (
            radial_velocity
            * np.sin(azimuth)
        )

        x = np.array(
            [
                px,
                py,
                vx,
                vy
            ],
            dtype=float
        )

        if R_measurement is None:

            R_measurement = self.R

        R_measurement = np.asarray(
            R_measurement,
            dtype=float
        )

        if R_measurement.shape != (3, 3):

            raise ValueError(
                "Measurement R must be "
                "a 3x3 covariance matrix."
            )

        sigma_range = np.sqrt(
            max(
                R_measurement[0, 0],
                1e-12
            )
        )

        sigma_angle = np.sqrt(
            max(
                R_measurement[1, 1],
                1e-12
            )
        )

        sigma_doppler = np.sqrt(
            max(
                R_measurement[2, 2],
                1e-12
            )
        )

        sigma_position = np.sqrt(
            sigma_range**2
            +
            (r * sigma_angle)**2
        )

        sigma_velocity = max(
            self.minimum_initial_velocity_sigma,
            self.initial_velocity_sigma_multiplier
            * sigma_doppler
        )

        P = np.diag([
            sigma_position**2,
            sigma_position**2,
            sigma_velocity**2,
            sigma_velocity**2
        ])

        track = Track(
            track_id=track_id,
            x=x,
            P=P,
            age=1,
            missed=0,
            hits=1,
            confirmed=(
                1 >= self.confirmation_hits
            )
        )

        self.tracks[track_id] = track

        self.tracks_ID[track_id] = 1

        return track

    # ========================================================
    # DELETE BAD TRACKS
    # ========================================================

    def delete_bad_tracks(self):

        for track_id in (
            self.get_active_track_indices()
        ):

            track = self.tracks[
                track_id
            ]

            if (
                track.missed
                > self.max_missed
            ):

                self.disable_track(
                    track_id
                )

    # ========================================================
    # EUCLIDEAN POSITION DISTANCE
    # ========================================================

    @staticmethod
    def euclidean_position_distance(
        track_i,
        track_j
    ):

        dx = (
            track_i.x[0]
            - track_j.x[0]
        )

        dy = (
            track_i.x[1]
            - track_j.x[1]
        )

        return np.hypot(
            dx,
            dy
        )

    # ========================================================
    # MERGE
    #
    # NO VELOCITY.
    #
    # ONLY:
    #
    # sqrt(
    #     (x1-x2)^2 +
    #     (y1-y2)^2
    # )
    # ========================================================

    def merge_close_tracks(
        self,
        distance_threshold=None,
        confirmed_only=None
    ):

        if distance_threshold is None:

            distance_threshold = (
                self.merge_distance_threshold
            )

        if confirmed_only is None:

            confirmed_only = (
                self.merge_confirmed_only
            )

        merge_found = True

        while merge_found:

            merge_found = False

            active_ids = (
                self.get_active_track_indices()
            )

            for a in range(
                len(active_ids)
            ):

                for b in range(
                    a + 1,
                    len(active_ids)
                ):

                    id_i = active_ids[a]
                    id_j = active_ids[b]

                    track_i = self.tracks[id_i]
                    track_j = self.tracks[id_j]

                    if (
                        track_i is None
                        or track_j is None
                    ):

                        continue

                    if confirmed_only:

                        if not (
                            track_i.confirmed
                            and track_j.confirmed
                        ):

                            continue

                    distance = (
                        self.euclidean_position_distance(
                            track_i,
                            track_j
                        )
                    )

                    if (
                        distance
                        > distance_threshold
                    ):

                        continue

                    # ----------------------------------------
                    # SAME SURVIVOR LOGIC AS YOUR ORIGINAL
                    # ----------------------------------------

                    if (
                        track_j.confirmed,
                        track_j.hits,
                        track_j.age
                    ) > (
                        track_i.confirmed,
                        track_i.hits,
                        track_i.age
                    ):

                        survivor = track_j
                        removed = track_i

                        survivor_id = id_j
                        removed_id = id_i

                    else:

                        survivor = track_i
                        removed = track_j

                        survivor_id = id_i
                        removed_id = id_j

                    # ----------------------------------------
                    # FUSION
                    # ----------------------------------------

                    P1_inv = np.linalg.pinv(
                        survivor.P
                    )

                    P2_inv = np.linalg.pinv(
                        removed.P
                    )

                    P_fused = np.linalg.pinv(
                        P1_inv + P2_inv
                    )

                    x_fused = (
                        P_fused
                        @ (
                            P1_inv
                            @ survivor.x
                            +
                            P2_inv
                            @ removed.x
                        )
                    )

                    survivor.x = x_fused

                    survivor.P = 0.5 * (
                        P_fused
                        + P_fused.T
                    )

                    survivor.hits = max(
                        survivor.hits,
                        removed.hits
                    )

                    survivor.age = max(
                        survivor.age,
                        removed.age
                    )

                    survivor.missed = min(
                        survivor.missed,
                        removed.missed
                    )

                    survivor.confirmed = (
                        survivor.confirmed
                        or removed.confirmed
                    )

                    self.disable_track(
                        removed_id
                    )

                    print(
                        "***************************************"
                        "Merge"
                        "********************************************************"
                    )

                    print(
                        f"MERGE: Track {removed_id} "
                        f"-> Track {survivor_id}, "
                        f"euclidean_distance="
                        f"{distance:.3f} m"
                    )

                    merge_found = True

                    break

                if merge_found:
                    break

    # ========================================================
    # GET STATES
    # ========================================================

    def get_track_states(self):

        result = []

        for track_id in (
            self.get_active_track_indices()
        ):

            track = self.tracks[
                track_id
            ]

            result.append({
                "id": track_id,
                "x": track.x.copy(),
                "P": track.P.copy(),
                "age": track.age,
                "hits": track.hits,
                "missed": track.missed,
                "confirmed": track.confirmed
            })

        return result

    # ========================================================
    # MASK
    # ========================================================

    def get_track_mask(self):

        return self.tracks_ID.copy()