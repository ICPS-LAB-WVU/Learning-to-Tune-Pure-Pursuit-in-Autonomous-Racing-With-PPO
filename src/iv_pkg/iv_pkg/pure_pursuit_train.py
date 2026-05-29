#!/usr/bin/env python3
import numpy as np
from scipy.spatial import distance, transform
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Float32


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.is_real = False
        self.map_name = 'Hockenheim_fast'
        self.is_ascending = True

        # --- RL-controlled values (with timestamps for fallback logic) ---
        self.L = 1.0
        self.steering_gain = 0.6
        self._last_ld_stamp = 0.0
        self._last_gain_stamp = 0.0
        self._rl_timeout = 0.75  # seconds since last message to treat as "stale"

        # Topics
        drive_topic = '/drive'
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        visualization_topic = '/visualization_marker_array'

        # Publishers & Subscribers
        self.sub_pose = self.create_subscription(
            PoseStamped if self.is_real else Odometry, odom_topic, self.pose_callback, 1
        )
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)

        # RL action topics
        self.sub_ld = self.create_subscription(Float32, '/lookahead_distance', self.lookahead_callback, 10)
        self.sub_gain = self.create_subscription(Float32, '/steering_gain', self.gain_callback, 10)
        self.sub_reset = self.create_subscription(Float32, '/episode_reset_signal', self.reset_callback, 10)

        self.drive_msg = AckermannDriveStamped()
        self.markerArray = MarkerArray()

        # Load waypoints (expects ; delimiter and header lines)
        map_path = os.path.abspath(os.path.join('src/iv_ws/src', 'csv_data'))
        csv_data = np.loadtxt(f'{map_path}/{self.map_name}.csv', delimiter=';', skiprows=3)
        self.waypoints = csv_data[:, 1:3]          # x, y
        self.ref_speed = csv_data[:, 5] * 1.3      # speed profile (scaled)
        self.curvatures = csv_data[:, 4]           # kappa_radpm
        self.numWaypoints = self.waypoints.shape[0]

        self.visualization_init()

    # === RL input callbacks ===
    def lookahead_callback(self, msg):
        self.L = float(np.clip(msg.data, 0.35, 4.0))
        self._last_ld_stamp = time.time()

    def gain_callback(self, msg):
        # reasonable bounds for gain
        self.steering_gain = float(np.clip(msg.data, 0.45, 1.15))
        self._last_gain_stamp = time.time()

    def reset_callback(self, msg):
        self.current_idx = 0

    # === Built-in linear teacher (used as SAFE fallback if RL signal is stale) ===
    def teacher_L_and_gain(self, speed, v_min, v_max):
        # L(v): linear
        d_min, d_max = 1.0, 2.5
        m_L = (d_max - d_min) / (v_max - v_min)
        b_L = d_min - m_L * v_min
        L = m_L * speed + b_L

        # Gain(v): linear (decreases with speed)
        gain_max, gain_min = 0.9, 0.65
        m_g = (gain_min - gain_max) / (v_max - v_min)
        b_g = gain_max - m_g * v_min
        K = m_g * speed + b_g

        return float(np.clip(L, 0.35, 4.0)), float(np.clip(K, 0.45, 1.15))

    def pose_callback(self, pose_msg):
        # --- pose/rotation ---
        if self.is_real:
            pos = pose_msg.pose
            quat = pose_msg.pose.orientation
        else:
            pos = pose_msg.pose.pose
            quat = pose_msg.pose.pose.orientation

        currX = pos.position.x
        currY = pos.position.y
        currPos = np.array([currX, currY]).reshape(1, 2)

        Rq = transform.Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        rot = Rq.as_matrix()

        # --- waypoint indexing ---
        distances = distance.cdist(currPos, self.waypoints, 'euclidean').reshape(self.numWaypoints)
        closest_index = int(np.argmin(distances))
        closestPoint = self.waypoints[closest_index]

        # --- reference speed from raceline ---
        speed = float(self.ref_speed[closest_index])

        # --- Fallback to teacher if RL actions are stale ---
        now = time.time()
        rl_ld_fresh = (now - self._last_ld_stamp) < self._rl_timeout
        rl_gain_fresh = (now - self._last_gain_stamp) < self._rl_timeout
        if not (rl_ld_fresh and rl_gain_fresh):
            v_min, v_max = 3.0, 18.0
            self.L, self.steering_gain = self.teacher_L_and_gain(speed, v_min, v_max)

        # --- find target point at distance >= L ---
        idx = int(closest_index)
        if self.is_ascending:
            while distances[idx] < self.L:
                idx = (idx + 1) % self.numWaypoints
        else:
            while distances[idx] < self.L:
                idx = (idx - 1 + self.numWaypoints) % self.numWaypoints
        targetPoint = self.waypoints[idx]

        # --- transform target into vehicle frame ---
        pv = (targetPoint - currPos).reshape(2)
        v_local = rot.T @ np.array([pv[0], pv[1], 0.0])
        y = v_local[1]

        # --- curvature/steering ---
        gamma = self.steering_gain * (2.0 * y / (self.L ** 2))
        gamma = float(np.clip(gamma, -0.35, 0.35))

        # --- publish drive (cap real-car speed) ---
        self.drive_msg.drive.steering_angle = gamma
        self.drive_msg.drive.speed = speed if not self.is_real else min(speed, 2.0)
        self.pub_drive.publish(self.drive_msg)

        # --- visualize ---
        self.targetMarker.points = [Point(x=targetPoint[0], y=targetPoint[1], z=0.0)]
        self.closestMarker.points = [Point(x=closestPoint[0], y=closestPoint[1], z=0.0)]
        self.markerArray.markers = [self.waypointMarker, self.targetMarker, self.closestMarker]
        self.pub_vis.publish(self.markerArray)

        print(f"Pos=({currX:.2f}, {currY:.2f}), Steer={gamma:.3f}, Speed={self.drive_msg.drive.speed:.2f}, "
              f"L={self.L:.2f}, gain={self.steering_gain:.2f}, RLfresh={(rl_ld_fresh and rl_gain_fresh)}")

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
        self.waypointMarker.points = [Point(x=w[0], y=w[1], z=0.0) for w in self.waypoints]
        self.targetMarker = create_marker(1, 'r', 0.2)
        self.closestMarker = create_marker(2, 'b', 0.2)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
