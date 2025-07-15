#!/usr/bin/env python3
import os, sys, timeit, math
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy


from acados_template import AcadosOcpSolver, AcadosSimSolver
from nmpc_tracker_ros2.mobile_robot_opt import MobileRobotOptimizer
from nmpc_tracker_ros2.mobile_robot_model import MobileRobotModel

HORIZON = 2
N_NODE  = 200
CTRL_RATE = 100.0   # Hz

class NMPCController(Node):
    def __init__(self):
        super().__init__('nmpc_controller')

        # 声明ROS参数
        self.declare_parameter('scale', 1)

        # --------------- ACADOS initialisation -----------------
        model = MobileRobotModel()
        self.opt = MobileRobotOptimizer(model.model,
                                        model.constraint,
                                        t_horizon=HORIZON,
                                        n_nodes=N_NODE)

        self.last_k = 0        # 上一周期的参考索引
        self.SEARCH = 100      # 向前最多搜索100个离散点（增加搜索范围）

        self.x_now = None   # will be filled after first odom msg
        self.tlog   = []

        # ROS pubs/subs
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 1)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 1)

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )


        self.sub_ref = self.create_subscription(Path, '/ref_path',
                                        self.ref_cb, qos)

        self.ref = None          # 等待接收
        self.ref_ready = False

        
        self.exec_path_pub = self.create_publisher(Path, '/exec_path', 10)
        self.pred_path_pub = self.create_publisher(Path, '/pred_path', 10)  # 新增：预测轨迹发布器

        self.exec_path_msg = Path()
        self.exec_path_msg.header.frame_id = 'odom'

        # timer: CTRL_RATE Hz
        self.timer = self.create_timer(1.0/CTRL_RATE, self.timer_cb)

    # ----------------------------------------------------------
    def odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        # yaw from quaternion
        th = math.atan2(2*(q.w*q.z + q.x*q.y),
                        1 - 2*(q.y*q.y + q.z*q.z))
        self.x_now = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            th
        ])

    # ----------------------------------------------------------
    def timer_cb(self):
        if not self.ref_ready or self.x_now is None:
            return  # waiting for first odom

        # 获取采样间隔参数
        SCALE = int(self.get_parameter('scale').value)

        # 前向搜索最近参考点
        search_end = min(self.last_k + self.SEARCH, len(self.ref) - 1)
        seg = self.ref[self.last_k : search_end + 1, :2]      # (M,2)
        dists = np.linalg.norm(seg - self.x_now[:2], axis=1)
        rel_idx = np.argmin(dists)
        k = self.last_k + rel_idx

        # 更新 last_k
        self.last_k = k
        
        ref = self.ref

        # --- rolling yref 填充（使用可调采样间隔）---
        for j in range(self.opt.N):
            idx = min(k + j * SCALE, len(ref) - 1)
            cos_r, sin_r = np.cos(ref[idx, 2]), np.sin(ref[idx, 2])
            yref = np.hstack((ref[idx, 0:2], cos_r, sin_r, np.zeros(self.opt.nu)))
            self.opt.solver.set(j, "yref", yref)

        idx_e = min(k + self.opt.N * SCALE, len(ref) - 1)
        cos_e, sin_e = np.cos(ref[idx_e, 2]), np.sin(ref[idx_e, 2])
        self.opt.solver.set(self.opt.N, "yref",
                            np.hstack((ref[idx_e, 0:2], cos_e, sin_e)))

        # --- current state硬约束 ---
        self.opt.solver.set(0, "lbx", self.x_now)
        self.opt.solver.set(0, "ubx", self.x_now)

        # --- solve NMPC ---
        tic = timeit.default_timer()
        status = self.opt.solver.solve()
        solve_t = timeit.default_timer() - tic
        self.tlog.append(solve_t)

        if status != 0:
            self.get_logger().warn(f"ACADOS status {status}")
            return

        u_now = self.opt.solver.get(0, "u")

        # --- publish cmd_vel ---
        msg = Twist()
        msg.linear.x  = float(u_now[0])
        msg.angular.z = float(u_now[1])
        self.pub_cmd.publish(msg)

        # --- 发布预测轨迹 ---
        self.publish_prediction_path()

        # --- log & publish executed path ---
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'odom'
        pose.pose.position.x = float(self.x_now[0])
        pose.pose.position.y = float(self.x_now[1])
        pose.pose.orientation.z = math.sin(self.x_now[2]/2.0)
        pose.pose.orientation.w = math.cos(self.x_now[2]/2.0)

        self.exec_path_msg.poses.append(pose)
        self.exec_path_pub.publish(self.exec_path_msg)


    def ref_cb(self, msg: Path):
        # 转换为 np.array  [N,3]
        poses = msg.poses
        if not poses:
            self.get_logger().warn("Received empty reference path")
            return

        data = np.zeros((len(poses), 3))
        for i, ps in enumerate(poses):
            data[i,0] = ps.pose.position.x
            data[i,1] = ps.pose.position.y
            # 取 yaw
            q = ps.pose.orientation
            data[i,2] = math.atan2(2*(q.w*q.z + q.x*q.y),
                                1 - 2*(q.y*q.y + q.z*q.z))
        self.ref = data
        self.ref_ready = True
        self.get_logger().info(f"Reference path received ({len(data)} pts)")


    def publish_prediction_path(self):
        """发布NMPC预测状态轨迹用于可视化"""
        pred_path = Path()
        t_now = self.get_clock().now().to_msg()
        pred_path.header.stamp = t_now
        pred_path.header.frame_id = 'odom'
        
        # 获取预测的状态序列
        for i in range(self.opt.N + 1):
            x_pred = self.opt.solver.get(i, "x")
            
            pose = PoseStamped()
            pose.header.stamp = t_now
            pose.header.frame_id = 'odom'
            pose.pose.position.x = float(x_pred[0])
            pose.pose.position.y = float(x_pred[1])
            pose.pose.position.z = 0.0
            
            # 从状态中的theta角度转换为四元数
            theta = float(x_pred[2])
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)
            
            pred_path.poses.append(pose)
        
        self.pred_path_pub.publish(pred_path)


def main():
    rclpy.init()
    node = NMPCController()
    rclpy.spin(node)

    # --------- after node shutdown: print stats ----------
    if node.tlog:
        t = np.array(node.tlog) * 1e3  # ms
        print(f"Solve time ⌀{t.mean():.2f} ms | "
              f"max {t.max():.2f} ms | min {t.min():.2f} ms")

if __name__ == '__main__':
    main()
