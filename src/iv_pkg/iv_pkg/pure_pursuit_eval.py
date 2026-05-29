#!/usr/bin/env python3
import os
import collections
import numpy as np
import rclpy
from scipy.spatial import distance, transform
from rclpy.node import Node

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped

import gym
from gym import spaces
import torch
from stable_baselines3 import PPO
import csv
import time


class PurePursuit(Node):
    """Pure Pursuit with PPO-selected (lookahead L, steering gain G). Speed comes from CSV."""
    def __init__(self):
        super().__init__('pure_pursuit_rl_eval')

        # === run mode ===
        self.is_real = False
        self.map_name = 'YasMarina_fast'
        self.is_ascending = True

        # Topics
        drive_topic = '/drive'
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        visualization_topic = '/visualization_marker_array'

        # Publishers & Subscribers
        self.sub_pose = self.create_subscription(
            PoseStamped if self.is_real else Odometry,
            odom_topic,
            self.pose_callback,
            1
        )
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)

        self.drive_msg = AckermannDriveStamped()
        self.markerArray = MarkerArray()
        self.k_offsets = [0, 5, 12]  # indices ahead used in training

        # === load CSV ===
        # Your original file used `csv = np.loadtxt(...)` which breaks `csv.writer`.
        # Keep it as csv_data to avoid shadowing the csv module.
        map_path = os.path.abspath(os.path.join('src/iv_ws/src', 'csv_data'))
        csv_data = np.loadtxt(
            os.path.join(map_path, f'{self.map_name}.csv'),
            delimiter=';',
            skiprows=3
        )
        self.waypoints = csv_data[:, 1:3]       # x,y
        self.curvatures = csv_data[:, 4]        # kappa
        self.ref_speed = csv_data[:, 5] * 1.172  # scaled command speed

        self.numWaypoints = self.waypoints.shape[0]

        # precompute path headings (optional)
        p0 = self.waypoints
        p1 = np.roll(self.waypoints, -1, axis=0)
        d = p1 - p0
        self.path_yaw = np.arctan2(d[:, 1], d[:, 0])

        # === state ===
        self.currX = 0.0
        self.currY = 0.0
        self.rot = np.eye(3)   # body->world rotation
        self.speed = 0.0
        self.current_wp_idx = 0

        # Last chosen action (for smoothing)
        self.prev_steering = 0.0
        self.prev_L = 1.5
        self.prev_G = 0.7
        self.L = 1.5
        self.G = 0.7

        # Action bounds (match training)
        self.L_MIN, self.L_MAX = 0.35, 4.0
        self.G_MIN, self.G_MAX = 0.45, 1.15

        # Safety heuristics
        self.straight_kappa_thresh = 0.02

        # histories
        self._idx_hist = collections.deque(maxlen=20)
        self._pos_hist = collections.deque(maxlen=20)

        # === RL model (2-D action [L, G]) ===
        candidate_paths = [
            '/sim_ws/src/iv_ws/src/ppo_lookahead_gain_model2.zip',
            '/mnt/data/ppo_lookahead_gain_model2.zip',
        ]
        model_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if model_path is None:
            raise FileNotFoundError("PPO model not found in expected locations; update candidate_paths.")

        self.model = PPO.load(model_path, env=self._make_dummy_env_2d())

        # VecNormalize stats (optional but recommended)
        self.obs_mean = None
        self.obs_var = None
        self._maybe_load_vecnorm_stats([
            './vecnorm2.pkl',
            '/mnt/data/vecnorm2.pkl',
            '/sim_ws/src/iv_ws/src/vecnorm2.pkl'
        ])

        # ---- Safety teacher takeover stats (for reviewer comment) ----
        self.total_steps = 0
        self.teacher_steps = 0
        self.teacher_events = 0
        self.teacher_max_streak_steps = 0
        self._teacher_streak_steps = 0
        self._prev_teacher_active = False

        # Timing (to convert streak steps -> seconds)
        self._last_cb_time = None
        self._dt_sum = 0.0
        self._dt_n = 0

        # "stale" watchdog (mainly useful if RL actions arrive via topic).
        # For local inference it should almost never trigger unless you want a watchdog.
        self.last_rl_time = time.time()
        self.RL_TIMEOUT = 0.50  # seconds

        self.visualization_init()
        self.get_logger().info(f"✅ Eval: PPO model loaded from {model_path} (L & G).")

        # --- Fig. 8 logging ---
        # In Docker as root, "~" expands to "/root"
        log_dir = os.path.expanduser("~/fig8_logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"policy_log_{self.map_name}.csv")

        self._log_f = open(self.log_path, "w", newline="")
        self._log_w = csv.writer(self._log_f)
        self._log_w.writerow([
            "t", "idx",
            "v_odom", "v_cmd",
            "k0", "k1", "k2", "dk", "kmax", "kappa_ahead",
            "L_rl", "G_rl", "L_used", "G_used",
            "teacher_active"
        ])
        self.get_logger().info(f"[Fig8] Logging to: {self.log_path}")

    # -------- teacher rule (linear fallback) --------
    def teacher_rule(self, v, kmax):
        # Match your paper’s teacher forms (adjust coefficients if needed)
        L_star = np.clip(0.50 + 0.28 * v - 3.5 * kmax, self.L_MIN, self.L_MAX)

        vmin, vmax = 3.0, 18.0
        gmax, gmin = 0.90, 0.65
        m = (gmin - gmax) / (vmax - vmin)
        b = gmax - m * vmin
        g_star = np.clip(m * v + b, self.G_MIN, self.G_MAX)

        return float(L_star), float(g_star)

    # -------- dummy eval env with 2-D action (L, G) --------
    def _make_dummy_env_2d(self):
        class EvalEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.action_space = spaces.Box(
                    low=np.array([0.35, 0.45], dtype=np.float32),
                    high=np.array([4.0, 1.15], dtype=np.float32),
                    dtype=np.float32
                )
                self.observation_space = spaces.Box(
                    low=np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
                    high=np.array([15.6, 0.5, 0.5, 0.5, 1.0], dtype=np.float32),
                    dtype=np.float32
                )

            def reset(self, **kwargs):
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            def step(self, action):
                return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, {}

        return EvalEnv()

    def _maybe_load_vecnorm_stats(self, try_paths):
        from stable_baselines3.common.vec_env import VecNormalize
        for p in try_paths:
            try:
                vec = VecNormalize.load(p)
                self.obs_mean = vec.obs_rms.mean.copy()
                self.obs_var = vec.obs_rms.var.copy()
                self.get_logger().info(f"Loaded VecNormalize stats: {p}")
                return
            except Exception:
                continue
        self.get_logger().warn("VecNormalize stats not found; continuing without obs normalization.")

    def _normalize_obs(self, obs):
        if self.obs_mean is None or self.obs_var is None:
            return obs
        return (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)

    # --------- helpers ----------
    def max_abs_curvature_ahead(self, arc_len=2.0, start_idx=None):
        if start_idx is None:
            start_idx = getattr(self, "closest_index", 0)

        avg_ds = float(np.mean(np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)))
        steps = 1 if avg_ds <= 1e-6 else max(1, int(arc_len / avg_ds))

        idxs = (start_idx + np.arange(steps)) % self.numWaypoints
        return float(np.max(np.abs(self.curvatures[idxs])))

    def get_lookahead_point(self, threshold):
        idx = self.closest_index
        while self.distances[idx] < threshold:
            idx = (idx + 1) % self.numWaypoints if self.is_ascending else (idx - 1 + self.numWaypoints) % self.numWaypoints
        return self.waypoints[idx]

    def transform_to_vehicle_frame(self, point):
        pvect = (point - np.array([self.currX, self.currY]))
        local3 = self.rot.T @ np.array([pvect[0], pvect[1], 0.0])
        return np.array([local3[0], local3[1], 0.0, 0.0])

    def curvature_features_eval(self):
        idx0 = self.closest_index
        n = self.numWaypoints
        k = self.curvatures
        idxs = [(idx0 + o) % n for o in self.k_offsets]
        k0, k1, k2 = [abs(float(k[i])) for i in idxs]
        dk = k1 - k0
        kmax = max(k0, k1, k2)
        return k0, k1, k2, dk, kmax

    # -------------- Odom --------------
    def pose_callback(self, msg):
        now = time.time()
        if self._last_cb_time is not None:
            dt = now - self._last_cb_time
            self._dt_sum += dt
            self._dt_n += 1
        self._last_cb_time = now

        # position, yaw, speed
        if self.is_real:
            pos = msg.pose
            quat = pos.orientation
        else:
            pos = msg.pose.pose
            quat = pos.orientation
            try:
                self.speed = float(msg.twist.twist.linear.x)
            except Exception:
                pass

        self.currX = pos.position.x
        self.currY = pos.position.y
        Rmat = transform.Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        self.rot = Rmat

        # closest waypoint
        currPos = np.array([[self.currX, self.currY]])
        self.distances = distance.cdist(currPos, self.waypoints).reshape((self.numWaypoints))
        self.closest_index = int(np.argmin(self.distances))
        self.closestPoint = self.waypoints[self.closest_index]
        speed_cmd = float(self.ref_speed[self.closest_index])

        # ---- RL observation ----
        k0, k1, k2, dk, kmax = self.curvature_features_eval()
        obs = np.array([self.speed, k0, k1, k2, dk], dtype=np.float32)
        obs_infer = self._normalize_obs(obs)

        # ---- decide RL vs teacher ----
        teacher_active = False
        action = None
        try:
            with torch.no_grad():
                action, _ = self.model.predict(obs_infer, deterministic=True)
            self.last_rl_time = time.time()
        except Exception as e:
            teacher_active = True
            self.get_logger().warn(f"[Teacher] RL inference failed -> teacher. err={e}")

        # Optional watchdog timeout (mostly relevant for topic-based RL)
        if (time.time() - self.last_rl_time) > self.RL_TIMEOUT:
            teacher_active = True

        # Compute (L_rl, G_rl)
        if teacher_active or action is None:
            L_rl, G_rl = self.teacher_rule(self.speed, kmax)
        else:
            arr = np.asarray(action).reshape(-1)
            if arr.size == 1:
                L_pred = float(arr[0])
                G_pred = self.prev_G
            else:
                L_pred = float(arr[0])
                G_pred = float(arr[1])

            L_rl = float(np.clip(L_pred, self.L_MIN, self.L_MAX))
            G_rl = float(np.clip(G_pred, self.G_MIN, self.G_MAX))

        # ---- update takeover stats ----
        self.total_steps += 1
        if teacher_active:
            self.teacher_steps += 1
            self._teacher_streak_steps += 1
            self.teacher_max_streak_steps = max(self.teacher_max_streak_steps, self._teacher_streak_steps)
        else:
            self._teacher_streak_steps = 0

        if teacher_active and (not self._prev_teacher_active):
            self.teacher_events += 1
        self._prev_teacher_active = teacher_active

        # Curvature-aware ceiling for L (protect hairpins)
        kappa_ahead = self.max_abs_curvature_ahead(arc_len=max(1.0, 1.5 * max(L_rl, 1.0)))
        L_cap = float(np.clip(1.3 / np.sqrt(kappa_ahead + 1e-6), self.L_MIN, 2.5))
        L = min(L_rl, L_cap)

        # Straight minimum to keep the car stable at higher speeds
        if self.speed >= 6.3 and kappa_ahead < self.straight_kappa_thresh:
            L_straight_min = float(np.clip(0.35 + 0.35 * self.speed, 1.5, 2.5))
            L = max(L, L_straight_min)

        # Action smoothing (reduce jitter)
        if kappa_ahead < self.straight_kappa_thresh:
            L = 0.6 * L + 0.4 * self.prev_L
            G = 0.6 * G_rl + 0.4 * self.prev_G
        else:
            L = 0.85 * L + 0.15 * self.prev_L
            G = 0.85 * G_rl + 0.15 * self.prev_G

        self.prev_L, self.prev_G = L, G
        self.L, self.G = L, G

        # Target & steering
        target = self.get_lookahead_point(self.L)
        translated = self.transform_to_vehicle_frame(target)
        y = translated[1]

        gamma = self.G * (2.0 * y / max(self.L, 1e-3) ** 2)
        gamma = float(np.clip(gamma, -0.35, 0.35))  # hard cap

        # Smooth steering angle
        alpha = 0.4
        gamma = (1 - alpha) * self.prev_steering + alpha * gamma
        self.prev_steering = gamma

        # --- Fig. 8 logging ---
        try:
            self._log_w.writerow([
                time.time(), self.closest_index,
                float(self.speed), float(speed_cmd),
                float(k0), float(k1), float(k2), float(dk), float(kmax), float(kappa_ahead),
                float(L_rl), float(G_rl), float(self.L), float(self.G),
                int(teacher_active)
            ])
            if (self.closest_index % 50) == 0:
                self._log_f.flush()
        except Exception as e:
            self.get_logger().warn(f"[Fig8] log failed: {e}")

        # publish command
        self.drive_msg.drive.steering_angle = gamma
        self.drive_msg.drive.speed = speed_cmd if not self.is_real else min(speed_cmd, 2.0)
        self.pub_drive.publish(self.drive_msg)

        # viz
        self.targetMarker.points = [Point(x=float(target[0]), y=float(target[1]), z=0.0)]
        self.closestMarker.points = [Point(x=float(self.closestPoint[0]), y=float(self.closestPoint[1]), z=0.0)]
        self.markerArray.markers = [self.waypointMarker, self.targetMarker, self.closestMarker]
        self.pub_vis.publish(self.markerArray)

        # logs
        self.get_logger().info(
            f"idx={self.closest_index:04d}, pos=({self.currX:.2f},{self.currY:.2f}), "
            f"L_rl={L_rl:.2f}, L_cap={L_cap:.2f}, L={self.L:.2f}, "
            f"G_rl={G_rl:.2f}, G={self.G:.2f}, "
            f"teacher={int(teacher_active)}, "
            f"steer={gamma:.2f}, v_csv={speed_cmd:.2f}, v_odom={self.speed:.2f}, "
            f"kappa_ahead={kappa_ahead:.5f}"
        )

    # -------------- viz --------------
    def visualization_init(self):
        def create_marker(marker_id, color, scale):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.type = Marker.POINTS
            marker.id = marker_id
            marker.scale.x = scale
            marker.scale.y = scale
            marker.color.a = 1.0
            setattr(marker.color, color, 0.75)
            return marker

        self.waypointMarker = create_marker(0, 'g', 0.05)
        self.waypointMarker.points = [Point(x=float(w[0]), y=float(w[1]), z=0.0) for w in self.waypoints]
        self.targetMarker = create_marker(1, 'r', 0.2)
        self.closestMarker = create_marker(2, 'b', 0.2)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Close CSV
        try:
            node._log_f.flush()
            node._log_f.close()
        except Exception:
            pass

        # Print teacher takeover stats
        rate = 100.0 * node.teacher_steps / max(1, node.total_steps)
        dt_est = (node._dt_sum / node._dt_n) if node._dt_n > 0 else 0.0
        max_streak_sec = node.teacher_max_streak_steps * dt_est if dt_est > 0 else 0.0

        node.get_logger().info(
            f"[TeacherStats] teacher_steps={node.teacher_steps}/{node.total_steps} "
            f"({rate:.3f}%), events={node.teacher_events}, "
            f"max_streak_steps={node.teacher_max_streak_steps}, "
            f"dt_est={dt_est:.4f}s, max_streak_sec={max_streak_sec:.3f}s"
        )

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
