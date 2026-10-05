#!/usr/bin/env python3
"""Read-only diagnostics for MRCLAM Robot1 SLAM.

Run from the project: python3 diagnose_slam.py
Ground truth is used ONLY for evaluation, never to modify SLAM inputs.
Uses the actual (possibly corrected) Barcodes.dat provided by the user.
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def find(root, name):
    for path in (root / name, root / "dataset" / name):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {name} in {root} or dataset/")


def read_dat(root, name):
    result = np.loadtxt(find(root, name), comments="#", ndmin=2)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"Invalid or empty {name}")
    return result


def report(label, errors, unit):
    errors = np.asarray(errors)
    errors = errors[np.isfinite(errors)]
    if not len(errors):
        print(f"{label}: no overlapping samples")
        return
    absolute = np.abs(errors)
    print(f"{label} (n={len(errors)}):")
    print(f"  RMSE={np.sqrt(np.mean(errors**2)):.4f} {unit}"
          f" | median abs={np.median(absolute):.4f}"
          f" | 95% abs={np.percentile(absolute, 95):.4f}"
          f" | max abs={absolute.max():.4f}")


def analyze(root):
    with find(root, "trajectory.csv").open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = ("timestamp", "odom_x", "odom_y", "odom_theta",
                  "slam_x", "slam_y", "slam_theta")
        records = np.array([[float(row[k]) for k in fields] for row in reader])
    if len(records) < 2 or not np.isfinite(records).all():
        raise ValueError("trajectory.csv needs at least two finite poses")
    t = records[:, 0]
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Trajectory timestamps must strictly increase")
    gt = read_dat(root, "Robot1_Groundtruth.dat")
    if np.any(np.diff(gt[:, 0]) <= 0):
        raise ValueError("Ground-truth timestamps must strictly increase")
    gt_unwrapped = np.unwrap(gt[:, 3])

    def pose_at(times):
        return np.column_stack([
            np.interp(times, gt[:, 0], gt[:, 1]),
            np.interp(times, gt[:, 0], gt[:, 2]),
            np.interp(times, gt[:, 0], gt_unwrapped),
        ])

    g = pose_at(t)
    valid = (t >= gt[0, 0]) & (t <= gt[-1, 0])
    intervals = valid[:-1] & valid[1:]
    gt_turn = np.diff(g[:, 2])
    # Odometry yaw is exported as a wrapped Pose2 angle. Exclude long
    # intervals with >pi true turn; their winding cannot be recovered here.
    intervals &= np.abs(gt_turn) < np.pi
    odom_turn = wrap(np.diff(records[:, 3]))
    slam_turn = wrap(np.diff(records[:, 6]))
    degrees = 180 / np.pi
    odom_error = wrap(odom_turn - gt_turn) * degrees
    slam_error = wrap(slam_turn - gt_turn) * degrees
    print("\nLOCAL ROTATION: same consecutive pose timestamps, independent of global drift")
    print(f"Pose intervals: median={np.median(dt):.3f}s, max={dt.max():.3f}s")
    report("Odometry rotation minus true rotation", odom_error[intervals], "deg")
    report("SLAM rotation minus true rotation", slam_error[intervals], "deg")
    print("\nLargest local SLAM rotation disagreements:")
    print("elapsed_s   dt_s   GT_turn_deg   ODOM_turn_deg   SLAM_turn_deg")
    indices = np.flatnonzero(intervals)
    ranked = indices[np.argsort(np.abs(slam_error[indices]))[::-1][:8]]
    for i in ranked:
        print(f"{t[i+1]-t[0]:9.3f} {dt[i]:7.3f} {gt_turn[i]*degrees:13.3f}"
              f" {odom_turn[i]*degrees:15.3f} {slam_turn[i]*degrees:15.3f}")

    # Heading is assessed relative to each trajectory's initial heading,
    # matching plot_slam.py's initial-pose alignment convention.
    print("\nHEADING after initial-pose alignment:")
    if not valid[0]:
        print("Initial timestamp outside GT coverage: heading comparison skipped.")
    else:
        gt_heading = g[:, 2] - g[0, 2]
        for label, column in (("Odometry heading error", 3), ("SLAM heading error", 6)):
            error = wrap(records[:, column] - records[0, column] - gt_heading) * degrees
            report(label, error[valid], "deg")

    commands = read_dat(root, "Robot1_Odometry.dat")
    print("\nRECORDED VELOCITY COMMANDS (not measured physical velocity):")
    print(f"Max |v command|={np.max(np.abs(commands[:, 1])):.5f} m/s")
    print(f"Max |omega command|={np.max(np.abs(commands[:, 2])):.5f} rad/s")

    measurements = read_dat(root, "Robot1_Measurement.dat")
    barcodes = read_dat(root, "Barcodes.dat")
    landmarks = read_dat(root, "Landmark_Groundtruth.dat")
    barcode_to_id = {int(row[1]): int(row[0]) for row in barcodes}
    if len(barcode_to_id) != len(barcodes):
        raise ValueError("Barcodes.dat contains duplicate barcode numbers")
    positions = {int(row[0]): row[1:3] for row in landmarks}
    ids = np.array([barcode_to_id.get(int(code), -1) for code in measurements[:, 1]])
    keep = np.array([subject in positions for subject in ids])
    keep &= (measurements[:, 0] >= gt[0, 0]) & (measurements[:, 0] <= gt[-1, 0])
    keep &= measurements[:, 2] > 0
    observations = measurements[keep]
    ids = ids[keep]
    if not len(observations):
        raise ValueError("No valid static-landmark observations overlap ground truth")
    robot = pose_at(observations[:, 0])
    landmark_xy = np.array([positions[subject] for subject in ids])
    delta = landmark_xy - robot[:, :2]
    predicted_range = np.linalg.norm(delta, axis=1)
    predicted_bearing = wrap(np.arctan2(delta[:, 1], delta[:, 0]) - robot[:, 2])
    range_errors = observations[:, 2] - predicted_range
    bearing_errors = wrap(observations[:, 3] - predicted_bearing) * degrees
    print("\nRAW LANDMARK OBSERVATIONS versus ground-truth geometry:")
    print("Residuals may include sensor, timestamp, extrinsic or ID errors.")
    report("Range residual", range_errors, "m")
    report("Bearing residual", bearing_errors, "deg")
    print("\nPer landmark (ID matched using YOUR Barcodes.dat):")
    print("ID  count   range_RMSE_m   bearing_RMSE_deg")
    for subject in sorted(set(ids)):
        mask = ids == subject
        print(f"{subject:2d} {np.count_nonzero(mask):6d}"
              f" {np.sqrt(np.mean(range_errors[mask]**2)):14.4f}"
              f" {np.sqrt(np.mean(bearing_errors[mask]**2)):18.4f}")
    print("\nPer true distance:")
    for low, high in ((0, 1), (1, 2), (2, 3), (3, np.inf)):
        mask = (predicted_range >= low) & (predicted_range < high)
        if np.any(mask):
            report(f"Range residual at {low}-{high:g}m", range_errors[mask], "m")
    return odom_error[intervals], slam_error[intervals], range_errors, bearing_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    analyze(args.root.resolve())


if __name__ == "__main__":
    main()
