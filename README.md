# VIDETEC2 — Radar Multi-Target Tracking

A radar-based multi-target tracking system using **DBSCAN**, **Extended Kalman Filtering (EKF)**, and **Joint Probabilistic Data Association (JPDA)**.

## Tracking Pipeline

```text
┌─────────────────────────┐
│   Radar Measurements    │
│ [Range, Azimuth, Doppler]│
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│    DBSCAN Clustering    │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ Cluster Mean & Covariance│
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│     EKF Prediction      │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  Radar Model & Jacobian │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   Gating & Likelihood   │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│    JPDA Association     │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│     EKF/JPDA Update     │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│    Track Management     │
│  Confirm / Delete / Merge│
└─────────────────────────┘
```

## Project Structure

```text
VIDETEC2-Radar-JPDA-Tracker/
│
├── main.ipynb
├── config.py
├── customed_dbscan.py
├── ekf.py
├── jpda.py
├── models.py
├── motion_model.py
├── radar_jpda_tracker.py
├── radar_model.py
└── track_manager.py
```

| File | Description |
|---|---|
| `main.ipynb` | Demo notebook for running and visualizing the tracker |
| `config.py` | Tracker parameters and measurement noise configuration |
| `customed_dbscan.py` | DBSCAN clustering of radar measurements |
| `ekf.py` | EKF prediction step |
| `jpda.py` | JPDA gating, event generation, and association probabilities |
| `models.py` | Track data structure |
| `motion_model.py` | Constant-velocity motion model |
| `radar_jpda_tracker.py` | Main radar JPDA-EKF tracking pipeline |
| `radar_model.py` | Radar measurement model and Jacobian |
| `track_manager.py` | Track initialization, confirmation, deletion, and merging |

## State & Measurements

### Track State

```text
x = [px, py, vx, vy]ᵀ
```

where:

- `px, py` — target position
- `vx, vy` — target velocity

### Radar Measurement

```text
z = [range, azimuth, radial_velocity]ᵀ
```

## Demo

Open **`main.ipynb`** to run the complete tracking example and visualize the estimated target tracks.

## Configuration

Main tracker parameters are defined in `TrackerConfig`:

```python
config = TrackerConfig(
    sigma_range=0.3,
    sigma_angle=np.deg2rad(1.0),
    sigma_doppler=0.17,
    pd=0.90,
    pg=0.99,
    max_missed=6,
    confirmation_hits=4
)
```
