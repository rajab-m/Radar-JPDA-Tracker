import numpy as np
from sklearn.cluster import DBSCAN


def radar_to_cartesian(measurements):
    measurements = np.asarray(measurements, dtype=float)
    r = measurements[:, 0]
    azimuth = measurements[:, 1]
    x = r * np.cos(azimuth)
    y = r * np.sin(azimuth)
    return np.column_stack([x, y])


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def mean_measurement(cluster_points):
    cluster_points = np.asarray(cluster_points, dtype=float)
    mean_range = np.mean(cluster_points[:, 0])
    sin_azimuth = np.mean(np.sin(cluster_points[:, 1]))
    cos_azimuth = np.mean(np.cos(cluster_points[:, 1]))
    mean_azimuth = np.arctan2(sin_azimuth, cos_azimuth)
    mean_doppler = np.mean(cluster_points[:, 2])
    return np.array([mean_range, mean_azimuth, mean_doppler])


def cluster_covariance(cluster_points, R_single):
    cluster_points = np.asarray(cluster_points, dtype=float)
    N = len(cluster_points)
    R_noise = R_single / N

    if N == 1:
        return R_noise

    z_bar = mean_measurement(cluster_points)
    residuals = cluster_points - z_bar
    residuals[:, 1] = wrap_angle(cluster_points[:, 1] - z_bar[1])

    sample_covariance = (residuals.T @ residuals) / (N - 1)
    R_spread = sample_covariance / N
    R_cluster = R_noise + R_spread
    R_cluster = 0.5 * (R_cluster + R_cluster.T)

    return R_cluster


def cluster_measurements(
    measurements,
    sigma_range=2.0,
    sigma_azimuth_deg=1.0,
    sigma_doppler=2.0,
    street_angle_deg=0.0,
    x_scale=2.0,
    y_scale=2.0,
    dbscan_eps=1.0,
    dbscan_min_samples=2
):
    """
    Cluster radar detections.

    street_angle_deg:
        Direction of the street relative to the radar x-axis.
        0° means the street follows the radar x-axis.
        Positive angles rotate the street counter-clockwise.

    x_scale:
        Clustering scale ALONG the street.

    y_scale:
        Clustering scale ACROSS the street.

    sigma_*:
        Single-detection measurement uncertainties.

    dbscan_eps:
        DBSCAN clustering radius after scaling.

    dbscan_min_samples:
        Minimum number of detections required for a cluster.
    """

    measurements = np.asarray(measurements, dtype=float)

    if len(measurements) == 0:
        return []

    # Single radar measurement covariance.
    sigma_azimuth = np.deg2rad(sigma_azimuth_deg)
    R_single = np.diag([
        sigma_range**2,
        sigma_azimuth**2,
        sigma_doppler**2
    ])

    # Convert radar coordinates to Cartesian coordinates.
    cartesian_measurements = radar_to_cartesian(measurements)
    x = cartesian_measurements[:, 0]
    y = cartesian_measurements[:, 1]

    # Rotate the coordinate system so that x = along street
    # and y = across street.
    street_angle = np.deg2rad(street_angle_deg)
    cos_angle = np.cos(street_angle)
    sin_angle = np.sin(street_angle)

    street_x = x * cos_angle + y * sin_angle
    street_y = -x * sin_angle + y * cos_angle

    street_coordinates = np.column_stack([street_x, street_y])

    # Scale the two directions independently before DBSCAN.
    scaled = street_coordinates / np.array([x_scale, y_scale])

    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels = db.fit_predict(scaled)

    clustered_measurements = []

    for label in sorted(set(labels)):

        if label == -1:
            noise_points = measurements[labels == label]

            for point in noise_points:
                clustered_measurements.append({
                    "z": point.copy(),
                    "R": R_single.copy()
                })

            continue

        cluster_points = measurements[labels == label]
        z_cluster = mean_measurement(cluster_points)
        R_cluster = cluster_covariance(cluster_points, R_single)

        clustered_measurements.append({
            "z": z_cluster,
            "R": R_cluster
        })

    return clustered_measurements