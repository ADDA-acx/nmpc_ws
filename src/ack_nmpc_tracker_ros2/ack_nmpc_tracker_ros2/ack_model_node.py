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
    简化前轮转向单车模型模拟器，支持加速度约束
    订阅 geometry_msgs/Twist @ /cmd:
        linear.x  -> v_target (目标速度 m/s)
        angular.z -> δ (前轮转角, rad)
    """
    def __init__(self):
        super().__init__('ackermann_twist_sim')

        # ---------------- 参数 ----------------
        self.declare_parameter('init_pose',   [0.0, 0.0, 0.0])  # x,y,theta
        self.declare_parameter('wheelbase',   1.0)              # m
        self.declare_parameter('max_steer',   0.6)              # rad (~34°)
        self.declare_parameter('max_accel',   0.2)              # m/s²
        self.declare_parameter('max_decel',   0.2)              # m/s²
        self.declare_parameter('cmd_topic',   '/cmd_vel')       # 可修改

        x0, y0, th0 = self.get_parameter('init_pose').value
        self.L        = float(self.get_parameter('wheelbase').value)
        self.max_steer= float(self.get_parameter('max_steer').value)
        self.max_accel= float(self.get_parameter('max_accel').value)
        self.max_decel= float(self.get_parameter('max_decel').value)
        cmd_topic     = self.get_parameter('cmd_topic').value

        # ---------------- 状态 ----------------
        self.state = np.array([x0, y0, th0], dtype=float)  # [x,y,theta]
        self.current_v = 0.0  # 当前实际速度
        self.target_v = 0.0   # 目标速度
        self.delta = 0.0      # 前轮转角

        # ---------------- 通信 ----------------
        self.sub = self.create_subscription(Twist, cmd_topic, self.cmd_cb, 1)
        self.odom_pub   = self.create_publisher(Odometry, '/odom', 1)
        self.marker_pub = self.create_publisher(Marker, '/robot_marker', 1)
        self.br         = TransformBroadcaster(self)

        self.last_t = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(DT, self.step)

        self.get_logger().info(
            f"AckermannTwistSim started. wheelbase={self.L:.3f} m, "
            f"max_steer={self.max_steer:.3f} rad, max_accel={self.max_accel:.3f} m/s², "
            f"max_decel={self.max_decel:.3f} m/s², cmd_topic='{cmd_topic}'")

    # ----------------------------------------------------
    def cmd_cb(self, msg: Twist):
        self.target_v = msg.linear.x
        delta = msg.angular.z
        # 限幅（避免 tan(delta) 爆炸）
        if delta > self.max_steer:
            delta = self.max_steer
        elif delta < -self.max_steer:
            delta = -self.max_steer
        self.delta = delta

    # ----------------------------------------------------
    def apply_acceleration_constraint(self, target_v, current_v, dt):
        """应用加速度约束"""
        v_diff = target_v - current_v
        
        if v_diff > 0:  # 加速
            max_v_change = self.max_accel * dt
        else:  # 减速
            max_v_change = self.max_decel * dt
        
        # 限制速度变化
        if abs(v_diff) > max_v_change:
            if v_diff > 0:
                new_v = current_v + max_v_change
            else:
                new_v = current_v - max_v_change
        else:
            new_v = target_v
            
        return new_v

    # ----------------------------------------------------
    def step(self):
        x, y, th = self.state

        # 时间步
        now = self.get_clock().now().nanoseconds * 1e-9
        dt  = now - self.last_t
        self.last_t = now
        if dt <= 0.0:
            return

        # 应用加速度约束
        self.current_v = self.apply_acceleration_constraint(self.target_v, self.current_v, dt)

        # --- 单车模型 ---
        x  += self.current_v * math.cos(th) * dt
        y  += self.current_v * math.sin(th) * dt
        th += (self.current_v / self.L) * math.tan(self.delta) * dt
        # wrap yaw (-pi, pi]
        th = math.atan2(math.sin(th), math.cos(th))
        self.state[:] = [x, y, th]

        yaw_rate = (self.current_v / self.L) * math.tan(self.delta)

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
        odom.twist.twist.linear.x  = float(self.current_v)
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
