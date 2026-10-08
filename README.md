# Radar-JPDA-Tracker
Radar-based multi-target tracking using DBSCAN clustering, Extended Kalman Filtering (EKF), and Joint Probabilistic Data Association (JPDA).

# VIDETEC2 — Radar Multi-Target Tracking

A radar-based multi-target tracking framework combining **DBSCAN measurement clustering**, a **Constant-Velocity Extended Kalman Filter (EKF)**, and **Joint Probabilistic Data Association (JPDA)**.

The tracker is designed for radar detections represented by:

```text
[range, azimuth, radial_velocity]
```

and maintains multiple target tracks while handling measurement noise, clutter, missed detections, ambiguous associations, track creation, confirmation, deletion, and merging.

---

## 1. System Overview

The overall processing pipeline is:

```text
                     Radar Detections
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Measurement Input   │
                 │ [r, θ, v_r]         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      DBSCAN         │
                 │ Measurement         │
                 │ Clustering          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Cluster Mean +      │
                 │ Measurement         │
                 │ Covariance          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    EKF Prediction   │
                 │ Constant Velocity   │
                 │ Model               │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Radar Measurement   │
                 │ Model + Jacobian    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Mahalanobis         │
                 │ Distance +          │
                 │ Likelihood          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   JPDA Gating       │
                 │ + Joint Events      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Marginal Association│
                 │ Probabilities β     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    EKF / JPDA       │
                 │      Update         │
                 └──────────┬──────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         New Tracks      Merge         Delete
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                    Active Track States
```

---

# 2. Main Components

The project is divided into the following modules:

```text
VIDETEC2/
│
├── main.py
├── config.py
├── radar-jpda-tracker.py
├── radar_model.py
├── motion_model.py
├── ekf.py
├── jpda.py
├── customed_dbscan.py
├── models.py
└── track_manager.py
```

### Module responsibilities

| Module | Purpose |
|---|---|
| `main.py` | Creates the tracker configuration and tracker instance |
| `config.py` | Central configuration of tracker parameters |
| `radar-jpda-tracker.py` | Main EKF + JPDA tracking pipeline |
| `radar_model.py` | Radar measurement model and Jacobian |
| `motion_model.py` | Constant-velocity state transition and process noise |
| `ekf.py` | EKF prediction |
| `jpda.py` | Gating, joint-event generation and JPDA probabilities |
| `customed_dbscan.py` | Radar measurement clustering and cluster covariance |
| `models.py` | Track data structure |
| `track_manager.py` | Track initialization, deletion, merging and state management |

---

# 3. State Representation

Each track uses a four-dimensional Cartesian state:

```text
x = [px, py, vx, vy]^T
```

where:

- `px` — target position along radar X
- `py` — target position along radar Y
- `vx` — target velocity along X
- `vy` — target velocity along Y

The radar measurement is three-dimensional:

```text
z = [r, θ, v_r]^T
```

where:

- `r` — range
- `θ` — azimuth
- `v_r` — radial velocity / Doppler

Therefore:

```text
State dimension       = 4
Measurement dimension = 3
```

---

# 4. Radar Measurement Model

The radar measurement model converts the Cartesian state into radar coordinates.

Given:

```text
x = [px, py, vx, vy]
```

the predicted radar measurement is:

```text
r = sqrt(px² + py²)

θ = atan2(py, px)

v_r = (px·vx + py·vy) / r
```

Implemented by:

```python
radar_measurement_model(x)
```

This produces:

```python
[r, azimuth, radial_velocity]
```

The corresponding nonlinear measurement Jacobian is calculated by:

```python
radar_measurement_jacobian(x)
```

The Jacobian is required by the EKF to linearize the radar measurement function around the current state.

---

# 5. Angle Handling

Radar azimuth is periodic, so a direct subtraction can produce an incorrect innovation.

For example:

```text
measurement = -179°
prediction  = +179°
```

A normal subtraction gives:

```text
-358°
```

although the actual angular difference is only:

```text
+2°
```

The project therefore normalizes the azimuth residual to:

```text
[-π, π)
```

using:

```python
wrap_to_pi(angle)
```

and:

```python
normalize_angle_residual(nu)
```

This is used during EKF/JPDA association calculations.

---

# 6. Motion Model

The tracker uses a **constant-velocity (CV)** motion model.

The state transition is:

```text
px(k+1) = px(k) + vx(k)·dt
py(k+1) = py(k) + vy(k)·dt

vx(k+1) = vx(k)
vy(k+1) = vy(k)
```

The state-transition matrix is:

```text
F =

[ 1   0   dt   0 ]
[ 0   1   0    dt]
[ 0   0   1    0 ]
[ 0   0   0    1 ]
```

implemented by:

```python
cv_state_transition(dt)
```

---

# 7. Process Noise

Acceleration uncertainty is incorporated through the process-noise covariance.

The model uses independent acceleration noise in X and Y:

```text
σ_ax
σ_ay
```

The resulting covariance contains the standard continuous white-noise acceleration terms:

```text
dt²
dt³
dt⁴
```

implemented by:

```python
cv_process_noise(
    dt,
    sigma_ax,
    sigma_ay
)
```

The EKF prediction is then:

```text
xₖ|ₖ₋₁ = F xₖ₋₁|ₖ₋₁

Pₖ|ₖ₋₁ = F P Fᵀ + Q
```

implemented by:

```python
ekf_predict(...)
```

---

# 8. EKF Prediction

Before processing a new radar frame, every active track is predicted forward in time.

For each active track:

```python
track.x, track.P = ekf_predict(
    track.x,
    track.P,
    dt,
    sigma_ax=config.sigma_ax,
    sigma_ay=config.sigma_ay
)
```

The timestamp difference is automatically calculated:

```text
dt = current_timestamp - previous_timestamp
```

For the first frame, `default_dt` is used.

Timestamps must be strictly increasing.

---

# 9. Measurement Clustering with DBSCAN

Radar sensors can generate multiple detections belonging to the same physical target.

The project provides a custom DBSCAN preprocessing module:

```python
cluster_measurements(...)
```

The process is:

```text
Radar detections
       │
       ▼
Polar → Cartesian
       │
       ▼
Rotate into street coordinates
       │
       ▼
Independent X/Y scaling
       │
       ▼
DBSCAN
       │
       ├── Clustered detections
       │
       └── Noise detections
```

---

# 10. Street-Aligned DBSCAN

The clustering implementation supports a street-oriented coordinate system.

The street direction is specified using:

```python
street_angle_deg
```

The measurements are rotated so that:

```text
street_x = direction along street
street_y = direction across street
```

The coordinates can then be independently scaled:

```python
x_scale
y_scale
```

This allows the clustering behavior to be different along and across the street.

For example:

```python
cluster_measurements(
    measurements,
    street_angle_deg=20.0,
    x_scale=2.0,
    y_scale=2.0,
    dbscan_eps=1.0,
    dbscan_min_samples=2
)
```

---

# 11. Cluster Measurement Representation

Each resulting detection is represented as:

```python
{
    "z": measurement,
    "R": covariance
}
```

where:

```text
z = [range, azimuth, radial_velocity]
```

and:

```text
R = 3 × 3 measurement covariance
```

This allows clustered measurements to carry their own uncertainty into the JPDA/EKF pipeline.

---

# 12. Cluster Covariance

For a cluster containing `N` radar detections, the measurement covariance consists of two components:

```text
R_cluster = R_noise + R_spread
```

The sensor noise contribution is reduced approximately according to:

```text
R_noise = R_single / N
```

The spatial/measurement spread of the cluster is calculated from the sample covariance and similarly scaled by `N`.

This means a cluster with tightly grouped detections can produce a relatively confident measurement, while a spatially dispersed cluster retains additional uncertainty.

Azimuth covariance is handled using wrapped angular residuals.

---

# 13. DBSCAN Noise Measurements

DBSCAN labels measurements that do not belong to a cluster as noise:

```text
label = -1
```

Noise measurements are **not discarded**.

Instead, each noise point becomes an individual radar measurement:

```python
{
    "z": point,
    "R": R_single
}
```

This allows isolated radar detections to continue into the tracking system.

---

# 14. JPDA

The core data-association algorithm is **Joint Probabilistic Data Association (JPDA)**.

JPDA is useful when multiple tracks and measurements are close enough that a hard nearest-neighbor assignment would be unreliable.

Instead of simply choosing:

```text
Track A → Measurement 1
```

the tracker calculates probabilities such as:

```text
Track A:
    missed detection       = 0.10
    Measurement 1          = 0.70
    Measurement 2          = 0.20
```

These probabilities are then used to update the track state probabilistically.

---

# 15. Mahalanobis Gating

For every track/measurement pair, the tracker calculates the innovation:

```text
ν = z - h(x)
```

with the azimuth component normalized.

The innovation covariance is:

```text
S = H P Hᵀ + R
```

The squared Mahalanobis distance is:

```text
D² = νᵀ S⁻¹ ν
```

The implementation avoids explicitly calculating `S⁻¹` where possible and instead solves the corresponding linear system.

---

# 16. Chi-Square Gate

JPDA uses a chi-square threshold based on the configured gate probability:

```python
gamma = chi2.ppf(
    pg,
    df=measurement_dimension
)
```

For the current implementation:

```text
measurement_dimension = 3
```

A measurement is inside the gate when:

```text
D² ≤ γ
```

The resulting Boolean matrix is:

```text
gate_matrix
```

with dimensions:

```text
number_of_tracks × number_of_measurements
```

---

# 17. Association Likelihood

For each track/measurement pair, the Gaussian likelihood is calculated from:

```text
D²
det(S)
measurement dimension
```

The log-likelihood is:

```text
log L =
    -0.5 D²
    -0.5 log(det(S))
    -0.5 m log(2π)
```

The likelihood is then:

```text
L = exp(log L)
```

This likelihood is passed to the JPDA module.

---

# 18. Joint Association Events

For every track, JPDA considers:

```text
-1 = missed detection
```

or one of the measurements inside its gate.

For example:

```text
Track 0 → measurement 0
Track 1 → measurement 1
```

is a valid joint event.

But:

```text
Track 0 → measurement 0
Track 1 → measurement 0
```

is invalid because one measurement cannot simultaneously be assigned to two tracks.

The implementation generates all valid combinations using:

```python
generate_joint_events(...)
```

---

# 19. Joint Event Probabilities

Each joint event receives a weight based on:

```text
Probability of detection
× measurement likelihood
÷ clutter density
```

Missed detections use:

```text
1 - Pd
```

The event weights are normalized to obtain joint probabilities.

These probabilities are then converted into marginal association probabilities.

---

# 20. Marginal Association Probabilities

The resulting `β` matrix has dimensions:

```text
number_of_tracks × (number_of_measurements + 1)
```

The first column represents:

```text
β_missed
```

and the remaining columns correspond to measurements.

For example:

```text
             Miss   M0    M1    M2
Track 0      0.10  0.70  0.15  0.05
Track 1      0.20  0.10  0.60  0.10
```

Each row represents the association probabilities for a track.

---

# 21. JPDA State Update

The tracker does not simply select the measurement with the highest probability.

Instead, the state innovation is calculated as a probability-weighted combination:

```text
ν̄ = Σ βᵢⱼ Kᵢⱼ νᵢⱼ
```

where:

- `βᵢⱼ` is the marginal association probability
- `Kᵢⱼ` is the Kalman gain
- `νᵢⱼ` is the innovation

The state is updated using:

```text
x ← x + ν̄
```

This allows multiple measurements to contribute to a single track update.

---

# 22. JPDA Covariance Update

The covariance update includes both:

1. Measurement covariance reduction
2. Association uncertainty

The implementation calculates the second moment of the possible state innovations and subtracts the squared mean innovation.

This produces an association covariance term.

The final covariance is:

```text
P_new =
    P
    - measurement_covariance_reduction
    + association_covariance
```

The matrix is symmetrized after the update.

---

# 23. Covariance Stabilization

Numerical errors can cause a covariance matrix to become slightly non-positive-semidefinite.

The implementation therefore performs an eigenvalue decomposition:

```python
eigenvalues, eigenvectors = np.linalg.eigh(P_new)
```

and applies:

```text
λᵢ ← max(λᵢ, covariance_eigenvalue_floor)
```

The covariance is reconstructed afterward.

This prevents singular or ill-conditioned covariance matrices from destabilizing subsequent calculations.

---

# 24. Track Management

Each track is represented by the `Track` dataclass:

```python
@dataclass
class Track:

    track_id: int
    x: np.ndarray
    P: np.ndarray

    age: int = 1
    missed: int = 0
    hits: int = 1
    confirmed: bool = False
```

Each track therefore stores:

```text
ID
State
Covariance
Age
Number of hits
Number of missed detections
Confirmation status
```

---

# 25. Track Initialization

A new radar measurement:

```text
[r, θ, v_r]
```

is converted into Cartesian position:

```text
px = r cos(θ)
py = r sin(θ)
```

and radial velocity is projected into Cartesian velocity:

```text
vx = v_r cos(θ)
vy = v_r sin(θ)
```

The initial state becomes:

```text
x = [px, py, vx, vy]
```

---

# 26. Initial Covariance

The initial position uncertainty combines range and angular uncertainty:

```text
σ_position² =
    σ_range²
    + (r σ_angle)²
```

Initial velocity uncertainty is:

```text
σ_velocity =
    max(
        minimum_initial_velocity_sigma,
        initial_velocity_sigma_multiplier
        × σ_doppler
    )
```

The initial state covariance is diagonal:

```text
P = diag(
    σ_position²,
    σ_position²,
    σ_velocity²,
    σ_velocity²
)
```

This prevents the initial velocity covariance from becoming unrealistically small when Doppler uncertainty is low.

---

# 27. Track Confirmation

New tracks begin with:

```text
hits = 1
```

A track becomes confirmed once:

```text
hits >= confirmation_hits
```

The default configuration uses:

```python
confirmation_hits = 4
```

Therefore a track generally requires repeated successful associations before being considered confirmed.

---

# 28. Hit / Miss Logic

After every JPDA update, the tracker calculates the total probability that the track was associated with a real measurement:

```text
association_probability =
    1 - β_missed
```

If:

```text
association_probability
>
association_probability_threshold
```

the track is considered a hit:

```text
missed = 0
hits += 1
```

Otherwise:

```text
missed += 1
```

The `association_probability_threshold` is therefore used for **track hit/miss management**, not for the JPDA probability calculation itself.

---

# 29. New Track Creation

The tracker can create a new track from a measurement when that measurement has insufficient association probability with existing tracks.

The configuration parameter is:

```python
measurement_probability
```

The logic is:

```text
measurement_probability =
    sum of association probabilities
    with existing tracks
```

A new track is created only when this value does not exceed the configured threshold.

With:

```python
measurement_probability = 0.0
```

the behavior is intentionally permissive:

> If a measurement falls inside a gate and has any positive association probability, it is considered associated with an existing track. A new track is created only when the measurement has zero association probability with all existing tracks.

Increasing `measurement_probability` makes new-track creation more restrictive.

---

# 30. Track Deletion

Tracks maintain a `missed` counter.

A track is deleted when:

```text
missed > max_missed
```

The default configuration is:

```python
max_missed = 6
```

This allows a track to survive several consecutive frames without a sufficiently strong association.

---

# 31. Maximum Number of Tracks

The track manager uses a fixed-size track array.

The default maximum is:

```python
max_tracks = 40
```

Track IDs range from:

```text
0 ... max_tracks - 1
```

Inactive slots are reused when new tracks are created.

---

# 32. Track Merging

The tracker supports merging tracks that are spatially too close.

The merge criterion is intentionally simple:

```text
Euclidean position distance
```

Only position is considered:

```text
d =
sqrt(
    (px₁ - px₂)² +
    (py₁ - py₂)²
)
```

Velocity is **not** used for the merge decision.

The configurable threshold is:

```python
merge_distance_threshold
```

For example:

```python
merge_distance_threshold = 2.0
```

means tracks within 2 meters are candidates for merging.

---

# 33. Merge Survivor Selection

When two tracks are merged, one track becomes the survivor.

The survivor is selected using:

```text
confirmed status
→ number of hits
→ track age
```

The better track according to these criteria is retained.

The removed track is then disabled.

---

# 34. Track Fusion

The two track estimates are fused using their covariance information.

For the survivor and removed tracks:

```text
P₁⁻¹
P₂⁻¹
```

are calculated using the pseudoinverse.

The fused covariance is:

```text
P_fused =
(P₁⁻¹ + P₂⁻¹)⁻¹
```

and the fused state is:

```text
x_fused =
P_fused
(
    P₁⁻¹ x₁
    +
    P₂⁻¹ x₂
)
```

The resulting state and covariance replace the survivor's previous estimate.

---

# 35. Complete Frame Processing

The main entry point is:

```python
tracker.step(
    measurements,
    timestamp
)
```

The processing sequence is:

### Step 1 — Validate measurements

Measurements are checked for:

- Correct dictionary structure
- Correct measurement shape
- Correct covariance shape
- Finite values
- Minimum range

Expected format:

```python
{
    "z": np.array([
        range,
        azimuth,
        radial_velocity
    ]),

    "R": np.array([
        [...],
        [...],
        [...]
    ])
}
```

---

### Step 2 — Calculate `dt`

For the first frame:

```text
dt = default_dt
```

For subsequent frames:

```text
dt =
current_timestamp
-
previous_timestamp
```

---

### Step 3 — First-frame initialization

If there are no active tracks, every valid measurement is used to initialize a track.

The tracker then performs track merging.

---

### Step 4 — Predict tracks

All active tracks are propagated using the constant-velocity EKF prediction.

---

### Step 5 — Handle missing measurements

If there are no measurements:

```text
missed += 1
age += 1
```

Bad tracks are subsequently deleted.

---

### Step 6 — Calculate association quantities

For every track/measurement pair, calculate:

```text
Predicted measurement
Jacobian
Innovation
Innovation covariance
Kalman gain
Mahalanobis distance
Likelihood
```

---

### Step 7 — JPDA

The tracker performs:

```text
Gating
    ↓
Joint event generation
    ↓
Joint event probabilities
    ↓
Marginal association probabilities
```

---

### Step 8 — EKF/JPDA update

Every active track is updated using its marginal association probabilities.

---

### Step 9 — Create new tracks

Measurements that are sufficiently unassociated with existing tracks can initialize new tracks.

---

### Step 10 — Merge tracks

Spatially close tracks are fused.

---

### Step 11 — Delete bad tracks

Tracks exceeding the missed-detection limit are removed.

---

### Step 12 — Return track states

The tracker returns the current active track states.

---

# 36. Configuration

The tracker is configured through:

```python
TrackerConfig
```

Example:

```python
config = TrackerConfig(
    sigma_range=0.3,
    sigma_angle=np.deg2rad(1.0),
    sigma_doppler=0.17,

    sigma_ax=1.0,
    sigma_ay=1.0,

    pd=0.90,
    pg=0.99,
    clutter_density=1e-4,

    association_probability_threshold=0.70,

    max_missed=6,
    confirmation_hits=4,
    min_range=0.2,
    max_tracks=40,

    measurement_probability=0.0,

    merge_distance_threshold=0.0,
    merge_confirmed_only=False,

    default_dt=0.1,

    minimum_initial_velocity_sigma=3.0,
    initial_velocity_sigma_multiplier=3.0,

    covariance_eigenvalue_floor=1e-10,

    debug=False
)
```

---

# 37. Configuration Parameters

## Radar Noise

| Parameter | Default | Description |
|---|---:|---|
| `sigma_range` | `0.3` | Range measurement standard deviation |
| `sigma_angle` | `1°` | Azimuth measurement standard deviation |
| `sigma_doppler` | `0.17` | Doppler/radial velocity standard deviation |

The measurement covariance is:

```python
R = np.diag([
    sigma_range**2,
    sigma_angle**2,
    sigma_doppler**2
])
```

---

## Motion Model

| Parameter | Default | Description |
|---|---:|---|
| `sigma_ax` | `1.0` | X-axis acceleration noise |
| `sigma_ay` | `1.0` | Y-axis acceleration noise |

Higher values allow the tracker to adapt more rapidly to changes in target velocity.

---

## JPDA

| Parameter | Default | Description |
|---|---:|---|
| `pd` | `0.90` | Probability of detection |
| `pg` | `0.99` | Probability covered by validation gate |
| `clutter_density` | `1e-4` | Clutter measurement density |

`pg` controls the statistical gating threshold.

---

## Track Management

| Parameter | Default | Description |
|---|---:|---|
| `association_probability_threshold` | `0.70` | Hit/miss decision threshold |
| `max_missed` | `6` | Maximum missed updates before deletion |
| `confirmation_hits` | `4` | Hits required for confirmation |
| `min_range` | `0.2` | Minimum valid radar range |
| `max_tracks` | `40` | Maximum simultaneous tracks |
| `measurement_probability` | `0.0` | Threshold controlling new-track creation |

---

## Track Merging

| Parameter | Default | Description |
|---|---:|---|
| `merge_distance_threshold` | `2.0` | Maximum Euclidean position distance |
| `merge_confirmed_only` | `False` | Merge only confirmed tracks when enabled |

The value used in `main.py` may differ from the dataclass default.

---

## Initialization

| Parameter | Default | Description |
|---|---:|---|
| `minimum_initial_velocity_sigma` | `5.0` | Minimum initial velocity uncertainty |
| `initial_velocity_sigma_multiplier` | `3.0` | Doppler uncertainty multiplier |

---

## Numerical Stability

| Parameter | Default | Description |
|---|---:|---|
| `covariance_eigenvalue_floor` | `1e-10` | Minimum covariance eigenvalue |

---

# 38. Basic Usage

A typical application creates the tracker once:

```python
import numpy as np

from config import TrackerConfig
from radar_jpda_tracker import RadarJPDAEKF

config = TrackerConfig(
    sigma_range=0.3,
    sigma_angle=np.deg2rad(1.0),
    sigma_doppler=0.17,

    sigma_ax=1.0,
    sigma_ay=1.0,

    pd=0.90,
    pg=0.99,
    clutter_density=1e-4,

    max_missed=6,
    confirmation_hits=4,
    max_tracks=40,

    default_dt=0.1,

    debug=False
)

tracker = RadarJPDAEKF(config)
```

Then process each radar frame:

```python
tracks = tracker.step(
    measurements,
    timestamp
)
```

---

# 39. Measurement Input Example

A measurement must contain both the radar observation and its covariance:

```python
measurements = [
    {
        "z": np.array([
            25.0,
            np.deg2rad(10.0),
            4.2
        ]),

        "R": np.diag([
            0.3**2,
            np.deg2rad(1.0)**2,
            0.17**2
        ])
    }
]
```

Multiple detections can be passed in the same frame:

```python
measurements = [
    {
        "z": np.array([25.0, np.deg2rad(10.0), 4.2]),
        "R": R1
    },
    {
        "z": np.array([32.0, np.deg2rad(-5.0), -2.1]),
        "R": R2
    }
]
```

---

# 40. Returned Track States

`tracker.step()` returns a list of active track states.

Each entry contains:

```python
{
    "id": track_id,
    "x": track.x.copy(),
    "P": track.P.copy(),
    "age": track.age,
    "hits": track.hits,
    "missed": track.missed,
    "confirmed": track.confirmed
}
```

The state vector is:

```text
x = [px, py, vx, vy]
```

Example:

```python
for track in tracker.step(measurements, timestamp):

    print("Track ID:", track["id"])
    print("Position:", track["x"][:2])
    print("Velocity:", track["x"][2:])
    print("Confirmed:", track["confirmed"])
```

---

# 41. Track Mask

The tracker also exposes:

```python
tracker.get_track_mask()
```

which returns the internal active/inactive track-slot mask.

For example:

```text
[1, 1, 0, 1, 0, 0, ...]
```

where:

```text
1 = active track
0 = inactive slot
```

---

# 42. Debug Mode

Debug output can be enabled with:

```python
debug=True
```

This provides information about:

- Track states
- Predicted measurements
- Measurement covariance
- Innovations
- Mahalanobis distances
- Gating matrix
- JPDA beta probabilities
- New track creation
- Track merging

For normal operation:

```python
debug=False
```

is recommended.

---

# 43. Important Design Characteristics

### Probabilistic association

Unlike nearest-neighbor tracking, JPDA allows several measurements to contribute probabilistically to one track.

### Nonlinear radar measurement

The radar observation is nonlinear because range, azimuth and radial velocity depend nonlinearly on Cartesian position and velocity.

### Measurement-specific covariance

Every measurement can carry its own covariance matrix. This is particularly useful after DBSCAN clustering because the uncertainty of a cluster can differ from the uncertainty of an individual radar detection.

### Position-only merging

Track merging deliberately ignores velocity. Two tracks are considered spatially close based only on:

```text
Euclidean distance in X/Y
```

### Explicit track lifecycle

Tracks progress through:

```text
Created
   ↓
Updated
   ↓
Confirmed
   ↓
Missed detections
   ↓
Deleted
```

with optional merging occurring during the lifecycle.

---

# 44. Numerical Considerations

The implementation takes several precautions against numerical instability:

- Symmetrization of covariance matrices
- Linear solves instead of explicit matrix inversion for innovation calculations
- Positive-definiteness checks using `slogdet`
- Covariance eigenvalue flooring
- Pseudoinverse during track fusion
- Range protection near the radar origin
- Azimuth residual normalization

These safeguards are particularly important because JPDA requires repeated evaluation of covariance matrices and likelihoods.

---

# 45. Computational Considerations

The JPDA implementation explicitly enumerates valid joint association events.

If there are:

```text
N tracks
M measurements
```

the number of possible events can grow rapidly as the number of gated track/measurement combinations increases.

Therefore, JPDA computational complexity can become significant in highly cluttered scenes with many simultaneously active tracks.

The gating stage is consequently important because it reduces the number of feasible associations before joint events are generated.

---

# 46. Recommended Processing Pipeline

If raw radar detections are available, the intended processing architecture is:

```python
clustered_measurements = cluster_measurements(
    raw_measurements,
    sigma_range=...,
    sigma_azimuth_deg=...,
    sigma_doppler=...,
    street_angle_deg=...,
    x_scale=...,
    y_scale=...,
    dbscan_eps=...,
    dbscan_min_samples=...
)

tracks = tracker.step(
    clustered_measurements,
    timestamp
)
```

This gives:

```text
Raw radar detections
        ↓
      DBSCAN
        ↓
Clustered radar measurements
        ↓
     RadarJPDAEKF
        ↓
   Active target tracks
```

---

# 47. Dependencies

The project uses the following Python packages:

```text
numpy
scipy
scikit-learn
```

Install them with:

```bash
pip install numpy scipy scikit-learn
```

A `requirements.txt` can contain:

```text
numpy
scipy
scikit-learn
```

---

# 48. Installation

Clone the repository:

```bash
git clone <repository-url>
cd VIDETEC2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then configure the tracker in `main.py` and connect the tracker to the radar measurement source.

---

# 49. File Structure

```text
VIDETEC2/
│
├── main.py
│
├── config.py
│   └── TrackerConfig
│
├── radar-jpda-tracker.py
│   └── RadarJPDAEKF
│
├── radar_model.py
│   ├── radar_measurement_model
│   ├── radar_measurement_jacobian
│   └── angle normalization
│
├── motion_model.py
│   ├── cv_state_transition
│   └── cv_process_noise
│
├── ekf.py
│   └── ekf_predict
│
├── jpda.py
│   └── JPDA
│
├── customed_dbscan.py
│   ├── radar_to_cartesian
│   ├── mean_measurement
│   ├── cluster_covariance
│   └── cluster_measurements
│
├── models.py
│   └── Track
│
└── track_manager.py
    └── TrackManager
```

---

# 50. Summary

VIDETEC2 combines several components into a complete radar multi-target tracking system:

```text
                  ┌───────────────┐
                  │ Radar Sensor  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │    DBSCAN     │
                  │   Clustering  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Measurement   │
                  │ + Covariance  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ EKF Prediction│
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │   Mahalanobis │
                  │    Gating     │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │     JPDA      │
                  │ Data Assoc.   │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ EKF/JPDA      │
                  │    Update     │
                  └───────┬───────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
          Create       Confirm      Merge
          Tracks        Tracks      Tracks
              │           │           │
              └───────────┼───────────┘
                          ▼
                  ┌───────────────┐
                  │ Track Manager │
                  │ Delete Stale  │
                  │    Tracks     │
                  └───────────────┘
```

The resulting tracker provides a probabilistic approach to multi-target radar tracking while explicitly modeling radar measurement uncertainty, nonlinear radar observations, clutter, missed detections, ambiguous associations, and track lifecycle management.
