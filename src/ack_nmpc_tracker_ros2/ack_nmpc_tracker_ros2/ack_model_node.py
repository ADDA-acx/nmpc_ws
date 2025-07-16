#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node


from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry

DT = 0.01   # 100 Hz

class AckermannTwistSim(Node):
    """
    简化前轮转向单车模型模拟器
    订阅 geometry_msgs/Twist @ /cmd:
        linear.x  -> v (m/s)
        angular.z -> δ (前轮转角, rad)
    """
    def __init__(self):
        super().__init__('ackermann_twist_sim')

        # ---------------- 参数 ----------------
        self.declare_parameter('init_pose',   [0.0, 0.0, 0.0])  # x,y,theta
        self.declare_parameter('wheelbase',   1.0)              # m
        self.declare_parameter('max_steer',   0.6)              # rad (~34°)
        self.declare_parameter('cmd_topic',   '/cmd_vel')           # 可修改

        x0, y0, th0 = self.get_parameter('init_pose').value
        self.L        = float(self.get_parameter('wheelbase').value)
        self.max_steer= float(self.get_parameter('max_steer').value)
        cmd_topic     = self.get_parameter('cmd_topic').value

        # ---------------- 状态 ----------------
        self.state = np.array([x0, y0, th0], dtype=float)  # [x,y,theta]
        self.last_cmd = np.zeros(2)  # [v, delta]

        # ---------------- 通信 ----------------
        self.sub = self.create_subscription(Twist, cmd_topic, self.cmd_cb, 1)
        self.odom_pub   = self.create_publisher(Odometry, '/odom', 1)
        self.marker_pub = self.create_publisher(Marker, '/robot_marker', 1)
        self.br         = TransformBroadcaster(self)

        self.last_t = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(DT, self.step)

        self.get_logger().info(
            f"AckermannTwistSim started. wheelbase={self.L:.3f} m, "
            f"max_steer={self.max_steer:.3f} rad, cmd_topic='{cmd_topic}'")

    # ----------------------------------------------------
    def cmd_cb(self, msg: Twist):
        v = msg.linear.x
        delta = msg.angular.z
        # 限幅（避免 tan(delta) 爆炸）
        if delta > self.max_steer:
            delta = self.max_steer
        elif delta < -self.max_steer:
            delta = -self.max_steer
        self.last_cmd[:] = [v, delta]

    # ----------------------------------------------------
    def step(self):
        v, delta = self.last_cmd
        x, y, th = self.state

        # 时间步
        now = self.get_clock().now().nanoseconds * 1e-9
        dt  = now - self.last_t
        self.last_t = now
        if dt <= 0.0:
            return

        # --- 单车模型 ---
        x  += v * math.cos(th) * dt
        y  += v * math.sin(th) * dt
        th += (v / self.L) * math.tan(delta) * dt
        # wrap yaw (-pi, pi]
        th = math.atan2(math.sin(th), math.cos(th))
        self.state[:] = [x, y, th]

        yaw_rate = (v / self.L) * math.tan(delta)

        # ---- TF ----
        stamp = self.get_clock().now().to_msg()
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id  = 'base_link'
        tf_msg.transform.translation.x = float(x)
        tf_msg.transform.translation.y = float(y)
        tf_msg.transform.rotation.z = math.sin(th/2.0)
        tf_msg.transform.rotation.w = math.cos(th/2.0)
        self.br.sendTransform(tf_msg)

        # ---- Odom ----
        odom = Odometry()
        odom.header = tf_msg.header
        odom.child_frame_id = tf_msg.child_frame_id
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.orientation = tf_msg.transform.rotation
        odom.twist.twist.linear.x  = float(v)
        odom.twist.twist.angular.z = float(yaw_rate)
        self.odom_pub.publish(odom)

        # ---- Marker ----
        self.publish_robot_marker(stamp, x, y, th)

    # ----------------------------------------------------
    def publish_robot_marker(self, stamp, x, y, th):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = 'odom'
        marker.ns = 'robot'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.1
        marker.pose.orientation.z = math.sin(th/2.0)
        marker.pose.orientation.w = math.cos(th/2.0)

        marker.scale.x = 1.0  # length
        marker.scale.y = 0.5  # width
        marker.scale.z = 0.2  # height

        marker.color.r = 0.0
        marker.color.g = 0.6
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)


def main():
    rclpy.init()
    node = AckermannTwistSim()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
