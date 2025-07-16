#!/usr/bin/env python3
import math, time
from pathlib import Path

import numpy as np                    # ➊ 新增：给 build_reference_path 用
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# ---- 轨迹参数，可改为 ROS 参数或通过 launch 配置 ----------
TOTAL_TIME = 120.0        # s
HORIZON     = 2.0         # s
N_NODE      = 200
A           = 10.0        # “∞”字幅度
N_LOOP      = 1

# ------------------------------------------------------------
#  把 build_reference_path 直接写进本文件
# ------------------------------------------------------------
def build_reference_path(total_time: float, h: float,
                         A: float = 10.0, n_loop: int = 1, center_x: float = 0.0, center_y: float = 0.0):
    """生成以 (center_x, center_y) 为中心的 ∞ 形参考轨迹，返回 [x, y, θ] 列表."""
    t_end = 2 * np.pi * n_loop
    t = np.arange(0.0, t_end + 1e-6, step=t_end / (total_time / h))

    x = A * np.sin(t) + center_x
    y = A * np.sin(t) * np.cos(t) + center_y

    dx, dy = np.gradient(x, t), np.gradient(y, t)
    theta = np.arctan2(dy, dx)
    return np.vstack((x, y, theta)).T   # shape: (N, 3)

# ------------------------------------------------------------
#  Path 发布节点
# ------------------------------------------------------------
class RefPublisher(Node):
    def __init__(self):
        super().__init__('ref_pub')

        # 初始化状态
        self.ref = None
        self.path_generated = False
        
        # 订阅 odom 获取初始位置
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,   # 迟到订阅者也能收到
        )
        self.pub = self.create_publisher(Path, '/ref_path', qos)

        # 每 2 s 重发一次，保险起见
        self.timer = self.create_timer(2.0, self.publish_path)

    def odom_callback(self, msg):
        """第一次收到 odom 时生成参考路径"""
        if not self.path_generated:
            # 获取当前位置作为圆心
            center_x = msg.pose.pose.position.x
            center_y = msg.pose.pose.position.y
            
            # 计算离散步长 (要与 NMPC 节点一致)
            h = HORIZON / N_NODE
            self.ref = build_reference_path(TOTAL_TIME, h, A=A, n_loop=N_LOOP, 
                                          center_x=center_x, center_y=center_y)
            
            self.path_generated = True
            self.get_logger().info(f"参考路径已生成，圆心位置: ({center_x:.2f}, {center_y:.2f})")
            
            # 立即发布一次
            self.publish_path()

    # ------------------------------------------------------
    def publish_path(self):
        if self.ref is None:
            self.get_logger().warn("等待 odom 数据以生成参考路径...")
            return
        
        msg = Path()
        t_now = self.get_clock().now().to_msg()
        msg.header.stamp = t_now
        msg.header.frame_id = 'odom'

        for p in self.ref:
            pose = PoseStamped()
            pose.header.stamp = t_now
            pose.header.frame_id = 'odom'
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])

            theta = float(p[2])
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)

            msg.poses.append(pose)

        self.pub.publish(msg)
        self.get_logger().info(f"Reference path ({len(msg.poses)} pts) published")

# ------------------------------------------------------------
def main():
    rclpy.init()
    node = RefPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
