import numpy as np

from itertools import product
from scipy.stats import chi2


class JPDA:

    def __init__(
        self,
        pd=0.90,
        pg=0.99,
        clutter_density=1e-4,
        measurement_dimension=3
    ):

        self.pd = pd
        self.pg = pg
        self.clutter_density = clutter_density
        self.measurement_dimension = (
            measurement_dimension
        )

    # ========================================================
    # GATING
    # ========================================================

    def gate_measurements(
        self,
        mahalanobis_matrix
    ):

        gamma = chi2.ppf(
            self.pg,
            df=self.measurement_dimension
        )

        gate_matrix = (
            mahalanobis_matrix <= gamma
        )

        return gate_matrix, gamma

    # ========================================================
    # JOINT EVENTS
    # ========================================================

    def generate_joint_events(
        self,
        gate_matrix
    ):

        n_tracks, n_measurements = (
            gate_matrix.shape
        )

        if n_tracks == 0:
            return [tuple()]

        choices = []

        for i in range(n_tracks):

            track_choices = [-1]

            for j in range(n_measurements):

                if gate_matrix[i, j]:

                    track_choices.append(j)

            choices.append(track_choices)

        events = []

        for candidate in product(*choices):

            assigned_measurements = [
                m
                for m in candidate
                if m != -1
            ]

            if (
                len(assigned_measurements)
                != len(set(assigned_measurements))
            ):

                continue

            events.append(tuple(candidate))

        return events

    # ========================================================
    # EVENT WEIGHT
    # ========================================================

    def event_weight(
        self,
        event,
        likelihood_matrix
    ):

        weight = 1.0

        miss_probability = max(
            1.0 - self.pd,
            1e-12
        )

        for (
            track_index,
            measurement_index
        ) in enumerate(event):

            if measurement_index == -1:

                weight *= miss_probability

            else:

                likelihood = (
                    likelihood_matrix[
                        track_index,
                        measurement_index
                    ]
                )

                weight *= (
                    self.pd
                    * likelihood
                    / max(
                        self.clutter_density,
                        1e-15
                    )
                )

        return weight

    # ========================================================
    # JOINT PROBABILITIES
    # ========================================================

    def calculate_joint_probabilities(
        self,
        events,
        likelihood_matrix
    ):

        if len(events) == 0:

            return np.array([])

        weights = np.array([
            self.event_weight(
                event,
                likelihood_matrix
            )
            for event in events
        ])

        total = np.sum(weights)

        if (
            total <= 0.0
            or not np.isfinite(total)
        ):

            probabilities = np.ones(
                len(events)
            )

            probabilities /= len(events)

        else:

            probabilities = (
                weights / total
            )

        return probabilities

    # ========================================================
    # MARGINAL PROBABILITIES
    # ========================================================

    def marginal_probabilities(
        self,
        events,
        event_probabilities,
        n_tracks,
        n_measurements
    ):

        beta = np.zeros(
            (
                n_tracks,
                n_measurements + 1
            )
        )

        for (
            event,
            event_probability
        ) in zip(
            events,
            event_probabilities
        ):

            for (
                i,
                measurement_index
            ) in enumerate(event):

                if measurement_index == -1:

                    beta[i, 0] += (
                        event_probability
                    )

                else:

                    beta[
                        i,
                        measurement_index + 1
                    ] += event_probability

        return beta

    # ========================================================
    # COMPLETE ASSOCIATION
    # ========================================================

    def associate(
        self,
        mahalanobis_matrix,
        likelihood_matrix
    ):

        n_tracks, n_measurements = (
            mahalanobis_matrix.shape
        )

        gate_matrix, gamma = (
            self.gate_measurements(
                mahalanobis_matrix
            )
        )

        gated_likelihoods = (
            likelihood_matrix.copy()
        )

        gated_likelihoods[
            ~gate_matrix
        ] = 0.0

        events = (
            self.generate_joint_events(
                gate_matrix
            )
        )

        event_probabilities = (
            self.calculate_joint_probabilities(
                events,
                gated_likelihoods
            )
        )

        beta = (
            self.marginal_probabilities(
                events,
                event_probabilities,
                n_tracks,
                n_measurements
            )
        )

        return (
            beta,
            gate_matrix,
            gamma,
            events,
            event_probabilities
        )