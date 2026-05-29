#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
import gym
from gym import spaces
import numpy as np
import time
import random
import os
from tf_transformations import quaternion_from_euler
from stable_baselines3.common.logger import Logger


class F1TenthEnv(gym.Env):
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()

        super().__init__()
        self.node = rclpy.create_node('rl_lookahead_gain_env')

        # === Action Space: [lookahead, steering_gain] ===
        #   L in [0.35, 4.0] m,  gain in [0.45, 1.15]
        self.action_space = spaces.Box(
            low=np.array([0.35, 0.45], dtype=np.float32),
            high=np.array([4.0, 1.15], dtype=np.float32),
            dtype=np.float32
        )

        # Curvature taps ahead
        self.k_offsets = [0, 5, 12]

        # Observation: [speed, k0, k1, k2, dk]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([15.6, 0.5, 0.5, 0.5, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # === Load track CSV ===
        map_path = os.path.abspath(os.path.join('src/iv_ws/src', 'csv_data'))
        csv_data = np.loadtxt(f'{map_path}/Hockenheim_fast.csv', delimiter=';', skiprows=3)
        self.waypoints = csv_data[:, 1:3]
        self.ref_speed = csv_data[:, 5] * 1.3
        self.curvatures = csv_data[:, 4]
        self.numWaypoints = self.waypoints.shape[0]

        # === State ===
        self.speed = 0.0
        self.curvature = 0.0
        self.lookahead_distance = 1.0
        self.prev_lookahead = 1.0
        self.steering_gain = 0.7
        self.prev_gain = 0.7

        self.step_count = 0
        self.max_steps = 10000
        self.stalled_steps = 0
        self.stall_limit = 200
        self.collision = False
        self.current_wp_idx = 0
        self.last_wp_idx = 0
        self._ep_progress = 0
        self.log_every_n = 1000  # log once every 1000 env steps



        self.tb_logger: Logger = None
        self._ep_L_sum = 0.0
        self._ep_G_sum = 0.0
        self._ep_speed_sum = 0.0
        self._ep_steps = 0
        self._ep_collision = 0
        self._ep_lap = 0

        self.teacher_coef_L = 4.0
        self.teacher_coef_G = 2.0

        # === ROS I/O ===
        self.lookahead_pub = self.node.create_publisher(Float32, '/lookahead_distance', 10)
        self.gain_pub = self.node.create_publisher(Float32, '/steering_gain', 10)
        self.reset_pub = self.node.create_publisher(Float32, '/episode_reset_signal', 10)

        self.odom_sub = self.node.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.node.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)

    # --- Sensors/derived features ---
    def odom_callback(self, msg):
        self.speed = msg.twist.twist.linear.x

    def attach_logger(self, logger: Logger):
        self.tb_logger = logger

    def _log_step(self, L, G):
        self._ep_L_sum += L
        self._ep_G_sum += G
        self._ep_speed_sum += self.speed
        self._ep_steps += 1

    def scan_callback(self, msg):
        if len(msg.ranges) > 0 and min(msg.ranges) < 0.2:
            self.collision = True

    def curvature_features(self):
        idxs = [(self.current_wp_idx + o) % self.numWaypoints for o in self.k_offsets]
        ks = np.abs(self.curvatures[idxs])
        k0, k1, k2 = ks
        dk = k1 - k0
        kmax = float(np.max(ks))
        return k0, k1, k2, dk, kmax

    def _obs(self):
        k0, k1, k2, dk, _ = self.curvature_features()
        return np.array([self.speed, k0, k1, k2, dk], dtype=np.float32)

    def pose_callback(self, msg):
        if self.waypoints is None or len(self.waypoints) == 0:
            return
        car_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        dists = np.linalg.norm(self.waypoints - car_pos, axis=1)
        closest_idx = np.argmin(dists)
        if closest_idx > self.current_wp_idx:
            self.current_wp_idx = closest_idx

        # smoothed curvature
        alpha = 0.8
        raw_curvature = self.curvatures[self.current_wp_idx]
        self.curvature = alpha * self.curvature + (1 - alpha) * raw_curvature

    # --- Episode control ---
    def reset(self):
        self.lookahead_distance = 1.0
        self.prev_lookahead = 1.0
        self.steering_gain = 0.7
        self.prev_gain = 0.7

        self.step_count = 0
        self.collision = False
        self.stalled_steps = 0
        self.current_wp_idx = 0
        self.last_wp_idx = 0

        self.reset_car_position()

        # signal to controller
        self.reset_pub.publish(Float32(data=1.0))

        self.teacher_coef_L = max(1.5, self.teacher_coef_L * 0.995)
        self.teacher_coef_G = max(0.8, self.teacher_coef_G * 0.995)


        return self._obs()

    def reset_car_position(self):
        reset_pub = self.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'

        # bias starts toward sharper bends
        if random.random() < 0.70:
            candidates = np.where(np.abs(self.curvatures) > 0.06)[0]
            idx = int(random.choice(candidates)) if len(candidates) > 0 else random.randint(0, len(self.waypoints) - 2)
        else:
            idx = random.randint(0, len(self.waypoints) - 2)

        x, y = self.waypoints[idx]
        nx, ny = self.waypoints[(idx + 1) % len(self.waypoints)]
        theta = np.arctan2(ny - y, nx - x)

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        q = quaternion_from_euler(0, 0, theta)
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = q

        reset_pub.publish(msg)
        time.sleep(0.5)

    # --- Teacher functions for reward shaping ---
    @staticmethod
    def ideal_lookahead(speed, kmax):
        # your existing heuristic: grows with speed, shrinks with curvature
        L = 0.50 + 0.28 * speed - 3.5 * kmax
        return float(np.clip(L, 0.35, 4.0))

    @staticmethod
    def ideal_gain_from_speed(speed, v_min=3.0, v_max=18.0, g_max=0.9, g_min=0.65):
        # linear rule from your training file
        m = (g_min - g_max) / (v_max - v_min)
        b = g_max - m * v_min
        return float(np.clip(m * speed + b, 0.45, 1.15))

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        L_cmd = float(np.clip(action[0], 0.35, 4.0))
        G_cmd = float(np.clip(action[1], 0.45, 1.15))

        # light smoothing to avoid jitter
        beta_L = 0.2
        beta_G = 0.2
        L = beta_L * L_cmd + (1.0 - beta_L) * self.prev_lookahead
        G = beta_G * G_cmd + (1.0 - beta_G) * self.prev_gain

        self.lookahead_distance = L
        self.steering_gain = G
        self.prev_lookahead = L
        self.prev_gain = G

        # publish both to controller
        self.lookahead_pub.publish(Float32(data=L))
        self.gain_pub.publish(Float32(data=G))

        rclpy.spin_once(self.node, timeout_sec=0.1)

        # features & teachers
        k0, k1, k2, dk, kmax = self.curvature_features()
        L_star = self.ideal_lookahead(self.speed, kmax)
        G_star = self.ideal_gain_from_speed(self.speed)

        # ---- ROS console log every N steps ----
        # use the *next* step index without changing your counters
        step_no = self.step_count + 1
        if step_no % self.log_every_n == 0:
            self.node.get_logger().info(
                f"[Step {step_no}] "
                f"Speed={self.speed:.2f} m/s, Curv={self.curvature:.5f}, "
                f"IdealL={L_star:.2f}, ChosenL={L:.2f}, "
                f"IdealG={G_star:.2f}, ChosenG={G:.2f}"
            )
        # ---------------------------------------

        # --- Reward ---
        reward = 0.0

        # go fast, but safely shaped by other terms
        reward += 1.8 * self.speed

        # track teacher signals
        reward -= self.teacher_coef_L * abs(L - L_star)
        reward -= self.teacher_coef_G * abs(G - G_star)

        # discourage jitter
        reward -= 0.4 * abs(L - self.prev_lookahead)
        reward -= 0.25 * abs(G - self.prev_gain)

        # curvature-aware shaping
        reward -= 1.2 * abs(self.curvature)
        reward -= 2.0 * (L * kmax)            # long L near high curvature is risky
        if kmax > 0.08 and L <= (1.2 + 0.05 * self.speed):
            reward += 1.5                      # pre-shortening before bends

        # encourage longer L & moderate G on straights at higher speed
        if self.speed >= 5.5 and abs(self.curvature) < 0.02:
            L_min_straight = float(np.clip(0.5 + 0.35 * self.speed, 1.5, 4.0))
            if L >= L_min_straight:
                reward += 1.0
            else:
                reward -= 1.0
            # cap overly aggressive gain on straights
            if G > 0.95:
                reward -= 0.5

        # collision & crawling penalties
        if self.collision:
            reward -= 10.0
        if self.speed < 0.1:
            reward -= 0.5

        # waypoint progress reward
        delta_wp = self.current_wp_idx - self.last_wp_idx
        if delta_wp > 0:
            reward += 1.0 * delta_wp
            self._ep_progress += delta_wp
            self.last_wp_idx = self.current_wp_idx

        # in per-episode logging block
        #self.tb_logger.record("rollout/wp_progress", float(self._ep_progress))
        #self._ep_progress = 0

        # done conditions
        self.step_count += 1
        self.stalled_steps = self.stalled_steps + 1 if self.speed < 0.05 else 0

        done = (
            self.collision or
            self.stalled_steps >= self.stall_limit or
            self.step_count >= self.max_steps
        )

        # lap complete (wrapped around)
        if self.current_wp_idx < self.last_wp_idx:
            self.node.get_logger().info("Episode ended: Lap completed")
            reward += 20.0
            done = True

        if done:
            if self.collision:
                self.node.get_logger().info("Episode ended: Collision")
            elif self.stalled_steps >= self.stall_limit:
                self.node.get_logger().info("Episode ended: Vehicle stalled")
            elif self.step_count >= self.max_steps:
                self.node.get_logger().info("Episode ended: Max steps reached")
                reward += 20.0

        if done and self.tb_logger is not None and self._ep_steps > 0:
            self.tb_logger.record("rollout/L_mean", self._ep_L_sum / self._ep_steps)
            self.tb_logger.record("rollout/gain_mean", self._ep_G_sum / self._ep_steps)
            self.tb_logger.record("rollout/speed_mean", self._ep_speed_sum / self._ep_steps)
            self.tb_logger.record("rollout/collision", 1.0 if self.collision else 0.0)
            self.tb_logger.record("rollout/wp_progress", float(self._ep_progress))  # <-- add this
            # reset accumulators
            self._ep_L_sum = self._ep_G_sum = self._ep_speed_sum = 0.0
            self._ep_steps = 0
            self._ep_progress = 0  # <-- reset here


        reward = float(np.clip(reward, -30.0, 100.0))
        self._log_step(L, G)
        return self._obs(), reward, done, {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
