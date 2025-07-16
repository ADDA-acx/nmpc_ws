#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpc_node.py – NMPC controller for front-steer bicycle model
Controls:  u = [v, δ]  (linear speed, steering angle)
"""

import os, sys, timeit, math
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg     import Odometry, Path

from acados_template import AcadosOcpSolver, AcadosSimSolver
from .ack_robot_opt import MobileRobotOptimizer  # ← 已改为单车
from .ack_robot_model import MobileRobotModel      # ← 前轮单车

from scipy.spatial import cKDTree

# ------------------------- 参数 -------------------------
HORIZON    = 2.0      # [s]
N_NODE     = 200
CTRL_RATE  = 100.0    # [Hz]
DEFAULT_V_REF = 0.4   # m/s，可 ROS param 化

class NMPCController(Node):
    def __init__(self):
        super().__init__('nmpc_controller')

        # ---------- ACADOS ----------
        model      = MobileRobotModel()    # 单车模型
        self.opt   = MobileRobotOptimizer(model.model,
                                          model.constraint,
                                          t_horizon=HORIZON,
                                          n_nodes=N_NODE)

        self.x_now    = None               # 当前状态
        self.tlog     = []

        # ---------- ROS I/O ----------
        self.pub_cmd  = self.create_publisher(Twist, '/cmd_vel', 1)
        self.sub_odom = self.create_subscription(Odometry, '/odom',
                                                 self.odom_cb, 1)

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.sub_ref  = self.create_subscription(Path, '/ref_path',
                                                 self.ref_cb, qos)

        self.exec_path_pub = self.create_publisher(Path, '/exec_path', 10)
        self.pred_path_pub = self.create_publisher(Path, '/pred_path', 10)

        self.exec_path_msg = Path()
        self.exec_path_msg.header.frame_id = 'odom'

        self.ref_ready  = False

        # --- spatial table ---
        self.path_xy  = None        # (N,2) 轨迹离散点
        self.s_acc    = None        # 弧长累计
        self.yaw_ref  = None        # 航向角
        self.kdtree   = None        # KD-Tree

        # 循环定时器
        self.timer = self.create_timer(1.0/CTRL_RATE, self.timer_cb)

    @staticmethod
    def interp1d(xq, xs, ys):
        """xs 升序，返回线性插值值"""
        idx = np.searchsorted(xs, xq)
        if idx <= 0:  return ys[0]
        if idx >= len(xs): return ys[-1]
        x0,x1 = xs[idx-1], xs[idx]
        y0,y1 = ys[idx-1], ys[idx]
        t = (xq-x0)/(x1-x0+1e-9)
        return (1-t)*y0 + t*y1

    # -------------------------------------------------- 回调
    def odom_cb(self, msg: Odometry):
        q  = msg.pose.pose.orientation
        th = math.atan2(2*(q.w*q.z + q.x*q.y),
                        1 - 2*(q.y*q.y + q.z*q.z))
        self.x_now = np.array([msg.pose.pose.position.x,
                               msg.pose.pose.position.y,
                               th])

    def ref_cb(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn("empty path"); return
        N = len(msg.poses)
        xy  = np.zeros((N,2))
        yaw = np.zeros(N)
        for i,ps in enumerate(msg.poses):
            xy[i]  = [ps.pose.position.x, ps.pose.position.y]
            q = ps.pose.orientation
            yaw[i] = math.atan2(2*(q.w*q.z + q.x*q.y),
                                1-2*(q.y*q.y+q.z*q.z))
        # 弧长
        ds = np.hypot(np.diff(xy[:,0]), np.diff(xy[:,1]))
        s_acc = np.concatenate(([0.0], np.cumsum(ds)))

        self.path_xy = xy
        self.s_acc   = s_acc
        self.yaw_ref = yaw
        self.kdtree  = cKDTree(xy)

        self.ref_ready = True
        self.get_logger().info(
            f"Path loaded ({N} pts, L={s_acc[-1]:.1f} m)")

    # -------------------------------------------------- 主循环
    def timer_cb(self):
        if not self.ref_ready or self.x_now is None:
            return

        # --- 最近点 KD-Tree ---
        dist,k = self.kdtree.query(self.x_now[:2])
        s0     = self.s_acc[k]

        # 采样步长 ds = v_ref*dt
        dt_mpc = HORIZON / N_NODE
        ds     = max(DEFAULT_V_REF * dt_mpc, 0.01)

        # 滚动 yref
        for j in range(self.opt.N):
            s   = s0 + j*ds
            x   = self.interp1d(s, self.s_acc, self.path_xy[:,0])
            y   = self.interp1d(s, self.s_acc, self.path_xy[:,1])
            th  = self.interp1d(s, self.s_acc, self.yaw_ref)
            yref = np.array([x, y,
                             math.cos(th), math.sin(th),
                             DEFAULT_V_REF, 0.0])
            self.opt.solver.set(j, "yref", yref)

        s_e = s0 + self.opt.N*ds
        x_e = self.interp1d(s_e, self.s_acc, self.path_xy[:,0])
        y_e = self.interp1d(s_e, self.s_acc, self.path_xy[:,1])
        th_e= self.interp1d(s_e, self.s_acc, self.yaw_ref)
        self.opt.solver.set(self.opt.N, "yref",
                np.array([x_e, y_e, math.cos(th_e), math.sin(th_e)]))

        # --- 3. 初值约束 ---
        self.opt.solver.set(0, "lbx", self.x_now)
        self.opt.solver.set(0, "ubx", self.x_now)

        # --- 4. 求解 NMPC ---
        tic = timeit.default_timer()
        status = self.opt.solver.solve()
        self.tlog.append(timeit.default_timer() - tic)

        if status != 0:
            self.get_logger().warn(f"ACADOS status {status}")
            return

        v_cmd, delta_cmd = self.opt.solver.get(0, "u")

        # --- 5. 发布 /cmd_vel ---
        cmd_msg = Twist()
        cmd_msg.linear.x  = float(v_cmd)
        cmd_msg.angular.z = float(delta_cmd)     # ← 现在是"转向角"不是角速度
        self.pub_cmd.publish(cmd_msg)

        # --- 6. 预测轨迹与执行轨迹 ---
        self.publish_prediction_path()
        self.log_executed_path()

    # -------------------------------------------------- 可视化辅助
    def publish_prediction_path(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'
        for i in range(self.opt.N + 1):
            x_pred = self.opt.solver.get(i, "x")
            pose   = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.pose.position.x = float(x_pred[0])
            pose.pose.position.y = float(x_pred[1])
            theta = float(x_pred[2])
            pose.pose.orientation.z = math.sin(theta/2.0)
            pose.pose.orientation.w = math.cos(theta/2.0)
            path.poses.append(pose)
        self.pred_path_pub.publish(path)

    def log_executed_path(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'odom'
        pose.pose.position.x = float(self.x_now[0])
        pose.pose.position.y = float(self.x_now[1])
        pose.pose.orientation.z = math.sin(self.x_now[2]/2.0)
        pose.pose.orientation.w = math.cos(self.x_now[2]/2.0)
        self.exec_path_msg.poses.append(pose)
        self.exec_path_pub.publish(self.exec_path_msg)


# ------------------------- 主入口 -------------------------
def main():
    rclpy.init()
    node = NMPCController()
    rclpy.spin(node)

    if node.tlog:
        t_ms = np.array(node.tlog) * 1e3
        print(f"acados solve time  mean {t_ms.mean():.2f} ms "
              f"| max {t_ms.max():.2f} ms | min {t_ms.min():.2f} ms")

if __name__ == '__main__':
    main()
