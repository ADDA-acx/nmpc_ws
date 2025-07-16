#!/usr/bin/env python3
import math, time
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker

DT = 0.01   # 100 Hz internal update

class UnicycleSim(Node):
    def __init__(self):
        super().__init__('unicycle_sim')
        self.declare_parameter('init_pose', [0.0, 0.0, 0.0])
        x0, y0, th0 = self.get_parameter('init_pose').value
        self.state = np.array([x0, y0, th0], dtype=float)

        self.sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_cb, 1)

        self.pub = self.create_publisher(Odometry, '/odom', 1)
        self.marker_pub = self.create_publisher(Marker, '/robot_marker', 1)
        self.br  = TransformBroadcaster(self)

        self.last_t = self.get_clock().now().nanoseconds * 1e-9

        self.last_cmd = np.zeros(2)  # v, ω
        self.timer = self.create_timer(DT, self.step)

    # --------------------------------------------------------
    def cmd_cb(self, msg: Twist):
        self.last_cmd[:] = [msg.linear.x, msg.angular.z]

    # --------------------------------------------------------
    def step(self):
        v, omega = self.last_cmd
        x, y, th = self.state
        # simple Euler integration
        now = self.get_clock().now().nanoseconds * 1e-9
        dt  = now - self.last_t
        self.last_t = now

        x  += v * math.cos(th) * dt
        y  += v * math.sin(th) * dt
        th += omega * dt
        self.state[:] = [x, y, th]

        # publish tf
        t = self.get_clock().now().to_msg()
        xform = TransformStamped()
        xform.header.stamp = t
        xform.header.frame_id = 'odom'
        xform.child_frame_id = 'base_link'
        xform.transform.translation.x = float(x)
        xform.transform.translation.y = float(y)
        xform.transform.rotation.z = math.sin(th/2.0)
        xform.transform.rotation.w = math.cos(th/2.0)
        self.br.sendTransform(xform)

        # publish odom
        odom = Odometry()
        odom.header.stamp = t
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.orientation.z = xform.transform.rotation.z
        odom.pose.pose.orientation.w = xform.transform.rotation.w
        odom.twist.twist.linear.x  = float(v)
        odom.twist.twist.angular.z = float(omega)
        self.pub.publish(odom)

        # publish robot marker
        self.publish_robot_marker(t, x, y, th)

    def publish_robot_marker(self, stamp, x, y, th):
        """发布机器人方块可视化"""
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = 'odom'
        marker.ns = 'robot'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # 设置位置和方向
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.1  # 稍微抬高一点
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(th/2.0)
        marker.pose.orientation.w = math.cos(th/2.0)
        
        # 设置尺寸 (长x宽x高)
        marker.scale.x = 0.8  # 机器人长度
        marker.scale.y = 0.4  # 机器人宽度
        marker.scale.z = 0.2  # 机器人高度
        
        # 设置颜色 (蓝色)
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        # 设置生命周期
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0  # 永久存在
        
        self.marker_pub.publish(marker)


def main():
    rclpy.init()
    node = UnicycleSim()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
