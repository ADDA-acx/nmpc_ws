#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpc_node.py – NMPC controller for front-steer bicycle model
Controls:  u = [v, δ]  (linear speed, steering angle)
"""

import os, sys, timeit, math
from pathlib import Path
import numpy as np
from collections import deque

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
N_NODE     = 100
CTRL_RATE  = 50.0    # [Hz]
DEFAULT_V_REF = 0.8   # m/s，可 ROS param 化

# 新增：求解时间监控参数
TIMING_WINDOW_SIZE = 100   # 统计窗口大小
TIMING_PRINT_RATE = 0.5   # Hz，打印频率降低

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
        
        # 新增：运行时求解时间监控
        self.solve_times = deque(maxlen=TIMING_WINDOW_SIZE)  # 固定长度队列
        self.interp_times = deque(maxlen=TIMING_WINDOW_SIZE)  # 插值时间
        self.solve_count = 0
        self.last_print_time = self.get_clock().now()

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

        # 主循环定时器 - 高频控制
        self.timer = self.create_timer(1.0/CTRL_RATE, self.timer_cb)
        
        # 🚀 新增：Path 打包降频定时器
        self.pred_timer = self.create_timer(0.1, self.publish_prediction_path)  # 10 Hz
        self.exec_timer = self.create_timer(0.2, self.log_executed_path)        # 5 Hz
        
        # 定时打印求解时间统计 - 降频
        self.timing_timer = self.create_timer(1.0/TIMING_PRINT_RATE, self.print_timing_stats)

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
        # 🚀 日志精简：降为DEBUG级别
        self.get_logger().debug(
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
        ds     = DEFAULT_V_REF * dt_mpc  # 确保最小步长

        # 🚀 批量插值优化 - 向量化计算
        interp_tic = timeit.default_timer()
        
        # 计算所有采样点的弧长
        s_vec = s0 + ds * np.arange(self.opt.N + 1)  # shape = (N+1,)
        
        # 批量插值 - 一次性计算所有点
        x_vec = np.interp(s_vec, self.s_acc, self.path_xy[:,0])   # (N+1,)
        y_vec = np.interp(s_vec, self.s_acc, self.path_xy[:,1])   # (N+1,)
        th_vec = np.interp(s_vec, self.s_acc, self.yaw_ref)       # (N+1,)
        
        # 预计算三角函数
        cos_th = np.cos(th_vec)  # (N+1,)
        sin_th = np.sin(th_vec)  # (N+1,)
        
        interp_time = timeit.default_timer() - interp_tic
        self.interp_times.append(interp_time * 1000)  # 转换为毫秒
        
        # 🚀 快速设置 yref - 避免重复插值计算
        for j in range(self.opt.N):
            yref = np.array([x_vec[j], y_vec[j],
                             cos_th[j], sin_th[j],
                             DEFAULT_V_REF, 0.0])
            self.opt.solver.set(j, "yref", yref)

        # 终端参考
        self.opt.solver.set(self.opt.N, "yref",
                np.array([x_vec[-1], y_vec[-1], cos_th[-1], sin_th[-1]]))

        # --- 3. 初值约束 ---
        self.opt.solver.set(0, "lbx", self.x_now)
        self.opt.solver.set(0, "ubx", self.x_now)

        # --- 4. 求解 NMPC ---
        tic = timeit.default_timer()
        status = self.opt.solver.solve()
        solve_time = timeit.default_timer() - tic
        
        # 记录求解时间
        self.tlog.append(solve_time)
        self.solve_times.append(solve_time * 1000)  # 转换为毫秒
        self.solve_count += 1

        if status != 0:
            # 🚀 降为WARN级别，减少输出
            self.get_logger().warn(f"ACADOS status {status}")
            return

        v_cmd, delta_cmd = self.opt.solver.get(0, "u")

        # --- 5. 发布 /cmd_vel ---
        cmd_msg = Twist()
        cmd_msg.linear.x  = float(v_cmd)
        cmd_msg.angular.z = float(delta_cmd)     # ← 现在是"转向角"不是角速度
        self.pub_cmd.publish(cmd_msg)

        # 🚀 Path 打包已移到独立定时器，主循环不再处理

    def print_timing_stats(self):
        """定期打印NMPC求解时间统计"""
        if len(self.solve_times) < 5:  # 至少需要5个样本
            return
            
        solve_times_ms = np.array(self.solve_times)
        interp_times_ms = np.array(self.interp_times) if self.interp_times else np.array([0])
        current_time = self.get_clock().now()
        
        # 🚀 精简计算 - 只计算关键统计
        solve_mean = np.mean(solve_times_ms)
        solve_max = np.max(solve_times_ms)
        
        interp_mean = np.mean(interp_times_ms)
        total_mean = solve_mean + interp_mean
        
        # 计算实时频率
        time_diff = (current_time - self.last_print_time).nanoseconds * 1e-9
        recent_count = min(len(self.solve_times), int(CTRL_RATE * time_diff))
        actual_freq = recent_count / time_diff if time_diff > 0 else 0
        
        # 判断性能状态
        status_emoji = "🟢" if total_mean < 20.0 else "🟡" if total_mean < 50.0 else "🔴"
        
        # 🚀 精简日志 - 只显示核心信息
        self.get_logger().info(
            f"{status_emoji} NMPC Core: Solve {solve_mean:.1f}ms + Interp {interp_mean:.2f}ms "
            f"= {total_mean:.1f}ms | {actual_freq:.1f}Hz | Peak: {solve_max:.1f}ms"
        )
        
        # 性能警告
        if total_mean > 20.0:
            self.get_logger().warn(f"⚠️  Core timing: {total_mean:.1f}ms > 20ms limit")
        
        self.last_print_time = current_time

    # -------------------------------------------------- 可视化辅助（降频）
    def publish_prediction_path(self):
        """预测轨迹发布 - 10Hz 独立定时器"""
        if not self.ref_ready or self.x_now is None:
            return
            
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
        """执行轨迹记录 - 5Hz 独立定时器"""
        if self.x_now is None:
            return
            
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

    # 程序结束时打印完整统计
    if node.tlog:
        solve_ms = np.array(node.tlog) * 1e3
        interp_ms = np.array(node.interp_times) if node.interp_times else np.array([0])
        
        print(f"\n📊 Final NMPC Performance Summary:")
        print(f"   Total solves: {len(node.tlog)}")
        print(f"   🔧 ACADOS Solve:     Mean: {solve_ms.mean():.2f}ms | Max: {solve_ms.max():.2f}ms")
        print(f"   📐 Batch Interpolation: Mean: {interp_ms.mean():.2f}ms | Max: {interp_ms.max():.2f}ms")
        print(f"   🔄 Core Pipeline:    Mean: {(solve_ms.mean() + interp_ms.mean()):.2f}ms")
        print(f"   📊 Percentiles:      P95: {np.percentile(solve_ms, 95):.2f}ms | P99: {np.percentile(solve_ms, 99):.2f}ms")

if __name__ == '__main__':
    main()
    rclpy.init()
    node = NMPCController()
    rclpy.spin(node)

    # 程序结束时打印完整统计
    if node.tlog:
        solve_ms = np.array(node.tlog) * 1e3
        interp_ms = np.array(node.interp_times) if node.interp_times else np.array([0])
        path_ms = np.array(node.path_pack_times) if node.path_pack_times else np.array([0])
        
        print(f"\n📊 Final NMPC Performance Summary:")
        print(f"   Total solves: {len(node.tlog)}")
        print(f"   🔧 ACADOS Solve:     Mean: {solve_ms.mean():.2f}ms | Max: {solve_ms.max():.2f}ms | Min: {solve_ms.min():.2f}ms")
        print(f"   📐 Interpolation:    Mean: {interp_ms.mean():.2f}ms | Max: {interp_ms.max():.2f}ms")
        print(f"   📦 Path Packaging:   Mean: {path_ms.mean():.2f}ms | Max: {path_ms.max():.2f}ms")
        print(f"   🔄 Total Pipeline:   Mean: {(solve_ms.mean() + interp_ms.mean() + path_ms.mean()):.2f}ms")
        print(f"   📊 Percentiles:      P95: {np.percentile(solve_ms, 95):.2f}ms | P99: {np.percentile(solve_ms, 99):.2f}ms")

if __name__ == '__main__':
    main()
