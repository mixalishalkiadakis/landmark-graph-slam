#!/usr/bin/env python3
"""SLAM replay / online pose monitor. Requires numpy and matplotlib.

python plot_slam.py                       # final-estimate replay, 20x speed
python plot_slam.py --speed 5             # simulated seconds / real second
python plot_slam.py --static              # save PNG without opening a window
python plot_slam.py --live                # follow online_trajectory.csv

Replay shows the final optimized poses from trajectory.csv. Live shows the
history of currentPose() estimates recorded after each iSAM2 update; it does
not redraw retrospective corrections to old graph poses. Optional landmarks
are the final map and are hidden initially to keep the trajectory readable.
"""

import argparse
import csv
import io
import time
from pathlib import Path

import numpy as np


FIELDS = ("timestamp", "odom_x", "odom_y", "odom_theta",
          "slam_x", "slam_y", "slam_theta")
ROOT = Path(__file__).resolve().parent


def find_file(root, *names, required=True):
    for name in names:
        path = root / name
        if path.is_file():
            return path
    if required:
        raise FileNotFoundError(f"Missing in {root}: {' or '.join(names)}")
    return None


def read_dat(path, columns):
    rows = []
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            text = line.strip()
            if text and not text.startswith("#"):
                values = list(map(float, text.split()))
                if len(values) < columns:
                    raise ValueError(f"Invalid row in {path}: {text}")
                rows.append(values[:columns])
    if not rows:
        raise ValueError(f"Empty file: {path}")
    result = np.asarray(rows)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite data in {path}")
    return result


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        return [[float(row[name]) for name in FIELDS] for row in reader]


class CsvFollower:
    """Read only newly appended, newline-terminated records.

    Handles creation, truncation, replacement, and a partially written row.
    The C++ logger must write a header, end every row with '\n', and flush.
    """
    def __init__(self, path):
        self.path = path
        self.reset()

    def reset(self):
        self.offset = 0
        self.pending = b""
        self.header = None
        self.rows = []
        self.identity = None

    def poll(self):
        if not self.path.is_file():
            return False
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        reset = self.identity is not None and (
            identity != self.identity or stat.st_size < self.offset)
        if reset:
            self.reset()
        self.identity = identity
        with self.path.open("rb") as source:
            source.seek(self.offset)
            chunk = source.read()
            self.offset = source.tell()
        self.pending += chunk
        lines = self.pending.split(b"\n")
        self.pending = lines.pop()
        changed = reset
        for raw in lines:
            if not raw.strip():
                continue
            values = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
            if self.header is None:
                self.header = values
                missing = set(FIELDS) - set(values)
                if missing:
                    raise ValueError(f"Live CSV: missing columns {sorted(missing)}")
                continue
            if len(values) != len(self.header):
                raise ValueError("Malformed complete row in live CSV")
            row = dict(zip(self.header, values))
            self.rows.append([float(row[name]) for name in FIELDS])
            changed = True
        return changed


def transform(points, origin):
    c, s = np.cos(origin[2]), np.sin(origin[2])
    return (np.asarray(points) - origin[:2]) @ np.array([[c, -s], [s, c]])


class Data:
    def __init__(self, rows, groundtruth):
        values = np.asarray(rows, dtype=float)
        if values.ndim != 2 or not len(values) or values.shape[1] != len(FIELDS):
            raise ValueError("Trajectory has no complete poses")
        if not np.isfinite(values).all():
            raise ValueError("Trajectory contains nonfinite values")
        self.timestamps = values[:, 0]
        if np.any(np.diff(self.timestamps) < 0):
            raise ValueError("Trajectory timestamps are not in time order")
        self.elapsed = self.timestamps - self.timestamps[0]
        gt_t = groundtruth[:, 0]
        if np.any(np.diff(gt_t) <= 0):
            raise ValueError("Ground-truth timestamps must be strictly increasing")
        t0 = self.timestamps[0]
        self.initial_in_range = bool(gt_t[0] <= t0 <= gt_t[-1])
        self.gt_origin = np.array([
            np.interp(t0, gt_t, groundtruth[:, 1]),
            np.interp(t0, gt_t, groundtruth[:, 2]),
            np.interp(t0, gt_t, np.unwrap(groundtruth[:, 3])),
        ])
        self.slam_origin = values[0, 4:7]
        self.slam = transform(values[:, 4:6], self.slam_origin)
        self.odom = transform(values[:, 1:3], values[0, 1:4])
        gt_at_pose = np.column_stack([
            np.interp(self.timestamps, gt_t, groundtruth[:, 1]),
            np.interp(self.timestamps, gt_t, groundtruth[:, 2]),
        ])
        self.truth = transform(gt_at_pose, self.gt_origin)
        self.gt_dense = transform(groundtruth[:, 1:3], self.gt_origin)
        self.gt_times = gt_t
        self.valid = (self.timestamps >= gt_t[0]) & (self.timestamps <= gt_t[-1])
        self.error = np.linalg.norm(self.slam - self.truth, axis=1)
        self.error[~self.valid] = np.nan
        count = np.cumsum(self.valid)
        self.rmse = np.sqrt(np.divide(
            np.cumsum(np.where(self.valid, self.error ** 2, 0)), count,
            out=np.full(len(count), np.nan), where=count > 0))


def load_landmarks(root):
    truth_path = find_file(root, "dataset/Landmark_Groundtruth.dat",
                           "Landmark_Groundtruth.dat", "01-Landmark_Groundtruth.dat",
                           required=False)
    estimate_path = find_file(root, "landmarks.csv", required=False)
    truth = read_dat(truth_path, 3) if truth_path else np.empty((0, 3))
    estimate = np.empty((0, 3))
    if estimate_path:
        with estimate_path.open(newline="", encoding="utf-8-sig") as source:
            estimate = np.asarray([[int(row["landmark_id"]), float(row["x"]),
                                    float(row["y"])] for row in csv.DictReader(source)])
        estimate = estimate.reshape(-1, 3)
    return truth, estimate


class View:
    def __init__(self, plt, widgets, live, speed, landmarks):
        self.plt = plt
        self.live = live
        self.speed = speed
        self.data = None
        self.index = 0
        self.playing = True
        self.seeking = False
        self.last_tick = time.monotonic()
        self.cursor_time = 0.0
        self.landmarks = landmarks
        self.landmark_artists = []
        self.fig = plt.figure(figsize=(13, 8))
        grid = self.fig.add_gridspec(2, 2, width_ratios=[3, 1.5],
                                     height_ratios=[1, 1], left=.07, right=.97,
                                     bottom=.28, top=.9, wspace=.24, hspace=.3)
        self.ax = self.fig.add_subplot(grid[:, 0])
        self.info = self.fig.add_subplot(grid[0, 1])
        self.info.axis("off")
        self.err_ax = self.fig.add_subplot(grid[1, 1])
        self.fig.suptitle("SLAM online pose history" if live else "SLAM final-estimate replay",
                          fontsize=17, fontweight="bold")
        self.ax.set(xlabel="x [m]", ylabel="y [m]")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(alpha=.18)
        self.err_ax.set(xlabel="Elapsed dataset time [s]", ylabel="Position error [m]")
        self.err_ax.grid(alpha=.2)
        self.odom_line, = self.ax.plot([], [], "--", color="#D97706", alpha=.22,
                                       lw=1.3, label="Odometry", zorder=1)
        self.gt_line, = self.ax.plot([], [], color="black", alpha=.28,
                                     lw=1.5, label="Ground truth", zorder=2)
        self.slam_line, = self.ax.plot([], [], color="#1565C0", lw=2.1,
                                       label="SLAM", zorder=4)
        self.recent_line, = self.ax.plot([], [], color="#42A5F5", lw=3.2, zorder=5)
        self.slam_point, = self.ax.plot([], [], "o", color="#1565C0", ms=8, zorder=7)
        self.gt_point, = self.ax.plot([], [], "o", color="black", alpha=.55, ms=6, zorder=7)
        self.error_link, = self.ax.plot([], [], ":", color="#EF4444", lw=1.5, zorder=6)
        self.ax.plot([0], [0], "+", color="black", ms=10, label="Initial pose")
        self.ax.legend(loc="upper left", fontsize=9)
        self.error_line, = self.err_ax.plot([], [], color="#1565C0", lw=1.6)
        self.error_point, = self.err_ax.plot([], [], "o", color="#EF4444", ms=5)
        self.error_cursor = self.err_ax.axvline(0, color="black", alpha=.25)
        self.info_text = self.info.text(0, 1, "Waiting for online_trajectory.csv...",
                                        va="top", fontsize=11, linespacing=1.65)
        self.note = self.fig.text(.07, .025,
            "Current-pose estimates after each update; old graph poses are not revised here."
            if live else "Final optimized trajectory replay. Landmarks (L) show the final map, not map growth.",
            fontsize=9, color="#555555")
        self.slider = widgets.Slider(self.fig.add_axes([.14, .135, .72, .025]),
                                      "Time [s]", 0, 1, valinit=0)
        self.slider.on_changed(self.seek)
        self.button = widgets.Button(self.fig.add_axes([.14, .065, .12, .04]), "Pause")
        self.button.on_clicked(self.toggle)
        self.back_button = widgets.Button(self.fig.add_axes([.29, .065, .12, .04]), "< pose")
        self.back_button.on_clicked(lambda event: self.step(-1))
        self.next_button = widgets.Button(self.fig.add_axes([.44, .065, .12, .04]), "pose >")
        self.next_button.on_clicked(lambda event: self.step(1))
        self.landmark_button = widgets.Button(self.fig.add_axes([.59, .065, .18, .04]), "Landmarks (L)")
        self.landmark_button.on_clicked(self.toggle_landmarks)
        self.fig.text(.07, .205, "Space: pause   |   Left/Right: one pose   |   L: final landmarks",
                      fontsize=10, color="#555555")
        self.fig.canvas.mpl_connect("key_press_event", self.key)

    def set_data(self, data):
        first = self.data is None
        origin_changed = not first and not np.array_equal(self.data.gt_origin, data.gt_origin)
        self.data = data
        duration = max(float(data.elapsed[-1]), .001)
        self.slider.valmax = duration
        self.slider.ax.set_xlim(0, duration)
        self.err_ax.set_xlim(0, duration)
        finite = data.error[np.isfinite(data.error)]
        self.err_ax.set_ylim(0, max(.1, float(np.max(finite)) * 1.12) if len(finite) else 1)
        gt_mask = (data.gt_times >= data.timestamps[0]) & (data.gt_times <= data.timestamps[-1])
        self.ax.update_datalim(np.vstack([data.slam, data.odom, data.gt_dense[gt_mask], data.truth]))
        self.ax.autoscale_view()
        if first or origin_changed:
            self.build_landmarks()
        if self.live and self.playing:
            self.index = len(data.elapsed) - 1
        elif first:
            self.index = 0
        else:
            self.index = min(self.index, len(data.elapsed) - 1)
        self.draw(self.index)

    def build_landmarks(self):
        for artist in self.landmark_artists:
            artist.remove()
        self.landmark_artists = []
        truth, estimate = self.landmarks
        for rows, origin, marker, color, prefix in [
            (truth, self.data.gt_origin, "o", "#C026D3", "T"),
            (estimate, self.data.slam_origin, "^", "#008577", "E"),
        ]:
            if not len(rows):
                continue
            xy = transform(rows[:, 1:3], origin)
            artist = self.ax.scatter(xy[:, 0], xy[:, 1], marker=marker,
                                      s=55, color=color, alpha=.75, zorder=8, visible=False)
            self.landmark_artists.append(artist)
            for row, point in zip(rows, xy):
                self.landmark_artists.append(self.ax.annotate(
                    f"{prefix}{int(row[0])}", point, xytext=(4, 5 if prefix == "E" else -12),
                    textcoords="offset points", color=color, fontsize=8, visible=False))

    def draw(self, index):
        d = self.data
        if d is None:
            return
        self.index = int(np.clip(index, 0, len(d.elapsed) - 1))
        i = self.index
        self.cursor_time = float(d.elapsed[i])
        self.slam_line.set_data(d.slam[:i+1].T)
        self.odom_line.set_data(d.odom[:i+1].T)
        lo = np.searchsorted(d.gt_times, d.timestamps[0], side="right")
        hi = np.searchsorted(d.gt_times, d.timestamps[i], side="left")
        dense = np.vstack([d.truth[0], d.gt_dense[lo:hi], d.truth[i]])
        self.gt_line.set_data(dense.T)
        recent_start = np.searchsorted(d.elapsed, max(0, d.elapsed[i] - 10))
        self.recent_line.set_data(d.slam[recent_start:i+1].T)
        self.slam_point.set_data([d.slam[i, 0]], [d.slam[i, 1]])
        if d.valid[i]:
            self.gt_point.set_data([d.truth[i, 0]], [d.truth[i, 1]])
            self.error_link.set_data([d.slam[i, 0], d.truth[i, 0]],
                                      [d.slam[i, 1], d.truth[i, 1]])
            self.error_point.set_data([d.elapsed[i]], [d.error[i]])
        else:
            self.gt_point.set_data([], [])
            self.error_link.set_data([], [])
            self.error_point.set_data([], [])
        self.error_line.set_data(d.elapsed[:i+1], d.error[:i+1])
        self.error_cursor.set_xdata([d.elapsed[i], d.elapsed[i]])
        interval = d.elapsed[i] - d.elapsed[i-1] if i else 0
        finite = d.error[:i+1][np.isfinite(d.error[:i+1])]
        max_error = float(finite.max()) if len(finite) else float("nan")
        error_text = f"{d.error[i]:.3f} m" if d.valid[i] else "outside GT interval"
        self.info_text.set_text(
            f"Pose {i + 1:,} / {len(d.elapsed):,}\n"
            f"Elapsed time: {d.elapsed[i]:.2f} s\n"
            f"Timestamp: {d.timestamps[i]:.6f}\n"
            f"Previous pose interval: {interval:.3f} s\n\n"
            f"SLAM position error: {error_text}\n"
            f"Trajectory RMSE so far: {d.rmse[i]:.3f} m\n"
            f"Maximum error so far: {max_error:.3f} m\n\n"
            + ("Online estimates" if self.live else f"Replay speed: {self.speed:g}x"))
        self.seeking = True
        self.slider.set_val(self.cursor_time)
        self.seeking = False
        self.fig.canvas.draw_idle()

    def seek(self, value):
        if not self.seeking and self.data is not None:
            self.playing = False
            self.button.label.set_text("Play")
            i = np.searchsorted(self.data.elapsed, value, side="right") - 1
            self.draw(max(0, i))

    def toggle(self, event=None):
        self.playing = not self.playing
        if self.playing and self.data is not None and not self.live and self.index == len(self.data.elapsed)-1:
            self.draw(0)
        self.last_tick = time.monotonic()
        self.button.label.set_text("Pause" if self.playing else "Play")

    def step(self, delta):
        self.playing = False
        self.button.label.set_text("Play")
        if self.data is not None:
            self.draw(self.index + delta)

    def toggle_landmarks(self, event=None):
        visible = bool(self.landmark_artists) and not self.landmark_artists[0].get_visible()
        for artist in self.landmark_artists:
            artist.set_visible(visible)
        self.fig.canvas.draw_idle()

    def key(self, event):
        if event.key == " ":
            self.toggle()
        elif event.key == "right":
            self.step(1)
        elif event.key == "left":
            self.step(-1)
        elif event.key == "l":
            self.toggle_landmarks()

    def tick(self, follower, groundtruth):
        now = time.monotonic()
        dt = now - self.last_tick
        self.last_tick = now
        if follower is not None:
            if follower.poll() and follower.rows:
                self.set_data(Data(follower.rows, groundtruth))
            elif self.data is not None and self.playing:
                self.draw(len(self.data.elapsed)-1)
        elif self.data is not None and self.playing:
            self.cursor_time = min(self.data.elapsed[-1], self.cursor_time + dt * self.speed)
            i = np.searchsorted(self.data.elapsed, self.cursor_time, side="right") - 1
            target = self.cursor_time
            self.draw(max(0, i))
            self.cursor_time = target  # retain sub-pose time, including long gaps
            if target >= self.data.elapsed[-1]:
                self.playing = False
                self.button.label.set_text("Replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="Project directory containing CSVs and dataset/")
    parser.add_argument("--speed", type=float, default=20, help="Dataset seconds per real second")
    parser.add_argument("--static", action="store_true", help="Save slam_results.png without a GUI")
    parser.add_argument("--live", action="store_true", help="Follow online_trajectory.csv as C++ appends rows")
    parser.add_argument("--trajectory", type=Path, help="Override trajectory CSV path")
    args = parser.parse_args()
    if args.speed <= 0 or not np.isfinite(args.speed):
        parser.error("--speed must be finite and positive")
    if args.live and args.static:
        parser.error("--live and --static cannot be combined")
    import matplotlib
    if args.static:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets
    if not args.static and "agg" == matplotlib.get_backend().lower():
        raise RuntimeError("No GUI backend. Run in a desktop/WSLg terminal, or use --static. Try python3 -m tkinter to check Tk.")
    root = args.root.resolve()
    gt_path = find_file(root, "dataset/Robot1_Groundtruth.dat", "Robot1_Groundtruth.dat", "02-Robot1_Groundtruth.dat")
    gt = read_dat(gt_path, 4)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})
    view = View(plt, widgets, args.live, args.speed, load_landmarks(root))
    follower = None
    if args.live:
        path = args.trajectory or root / "online_trajectory.csv"
        follower = CsvFollower(path)
        print(f"Following {path}; waiting for flushed CSV rows.")
    else:
        path = args.trajectory or find_file(root, "trajectory.csv", "05-trajectory.csv")
        data = Data(read_rows(path), gt)
        view.set_data(data)
        if not data.initial_in_range:
            print("Warning: initial timestamp outside GT interval; origin uses closest GT endpoint. Out-of-range errors are excluded.")
        print(f"Final-trajectory position RMSE: {data.rmse[-1]:.6f} m")
        print("Error is Euclidean SLAM-to-GT distance at the SAME timestamp after initial-pose alignment.")
    if args.static:
        view.draw(len(view.data.elapsed)-1)
        output = root / "slam_results.png"
        view.fig.savefig(output, dpi=180)
        plt.close(view.fig)
        print(f"Saved {output}")
    else:
        # Timer callbacks remain alive for as long as the figure is open.
        timer = view.fig.canvas.new_timer(interval=200 if args.live else 40)
        timer.add_callback(view.tick, follower, gt)
        timer.start()
        plt.show()
        timer.stop()


if __name__ == "__main__":
    main()
