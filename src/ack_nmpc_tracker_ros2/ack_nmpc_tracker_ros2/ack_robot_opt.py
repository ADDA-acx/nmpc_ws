#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mobile_robot_opt.py – acados 0.5.x   NMPC optimizer
NONLINEAR_LS cost   (x, y, cosθ, sinθ, v, δ)
"""

import os, sys, shutil
from pathlib import Path
import numpy as np
import scipy.linalg
import casadi as ca

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from .ack_robot_model import MobileRobotModel   # 已替换成“单车”模型


def safe_mkdir_recursive(directory, overwrite=False):
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    elif overwrite:
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)


class MobileRobotOptimizer:
    """
    基于前轮转向单车模型的 NMPC 优化器
    controls: u = [v, δ]  (线速度, 前轮转向角)
    """

    def __init__(self, m_model, m_constraint, t_horizon: float, n_nodes: int):
        # ---------- 时域 & 维度 ----------
        self.T, self.N = float(t_horizon), int(n_nodes)
        self.nx = m_model.x.shape[0]
        self.nu = m_model.u.shape[0]
        dt     = self.T / self.N 

        a_max  = 0.2                           # 纵向加速度上限 m/s²
        d_max  = 0.2                           # 转角速率上限 rad/s

        # ---------- acados 路径 ----------
        acados_path = Path(os.environ["ACADOS_SOURCE_DIR"]).resolve()
        sys.path.insert(0, str(acados_path))
        safe_mkdir_recursive("./acados_models")

        # ---------- OCP ----------
        ocp = AcadosOcp()
        ocp.acados_include_path = str(acados_path / "include")
        ocp.acados_lib_path     = str(acados_path / "lib")
        ocp.model               = m_model
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf        = self.T
        ocp.dims.np = 2

        # ---------- COST ----------
        ocp.cost.cost_type   = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.cost_type_0 = "NONLINEAR_LS"

        # y = [x, y, cosθ, sinθ, v, δ]
        x_sym, y_sym, th_sym = m_model.x[0], m_model.x[1], m_model.x[2]
        v_sym, delta_sym     = m_model.u[0], m_model.u[1]
        p_sym            = m_model.p          # 就是 (v_prev, delta_prev)

        expr_y   = ca.vertcat(x_sym, y_sym, ca.cos(th_sym),
                              ca.sin(th_sym), v_sym, delta_sym)
        expr_y_e = ca.vertcat(x_sym, y_sym,
                              ca.cos(th_sym), ca.sin(th_sym))


        h_expr = ca.vertcat(
        v_sym     - p_sym[0],             # Δv
        delta_sym - p_sym[1]              # Δδ
        )

        ocp.model.cost_y_expr    = expr_y
        ocp.model.cost_y_expr_e  = expr_y_e
        ocp.model.cost_y_expr_0  = expr_y

        # 权重矩阵
        Q_pos = np.diag([5.0, 5.0])          # x, y
        Q_ang = np.diag([1.0, 1.0])          # cosθ, sinθ
        R     = np.diag([0.5, 2.0])         # v, δ

        W   = scipy.linalg.block_diag(Q_pos, Q_ang, R)   # 6×6
        W_e = scipy.linalg.block_diag(Q_pos, Q_ang)      # 4×4

        ocp.cost.W      = W
        ocp.dims.ny     = 6
        ocp.cost.yref   = np.zeros(6)

        ocp.cost.W_e    = W_e
        ocp.dims.ny_e   = 4
        ocp.cost.yref_e = np.zeros(4)

        ocp.cost.W_0    = W
        ocp.dims.ny_0   = 6
        ocp.cost.yref_0 = np.zeros(6)


        ocp.model.con_h_expr = h_expr         # 注册到 acados
        ocp.dims.nh          = 2              # 有两个 h

        # ---------- 控制约束 ----------
        ocp.constraints.x0    = np.zeros(self.nx)
        ocp.constraints.lbu   = np.array([m_constraint.v_min,
                                          m_constraint.delta_min])
        ocp.constraints.ubu   = np.array([m_constraint.v_max,
                                          m_constraint.delta_max])
        ocp.constraints.idxbu = np.array([0, 1])

        ocp.constraints.lh   = np.array([-a_max*dt, -d_max*dt])
        ocp.constraints.uh   = np.array([ a_max*dt,  d_max*dt])

        # ---------- 参数值初始化 ----------
        ocp.parameter_values = np.zeros(2)  # 初始化参数值 [v_prev, delta_prev]

        # ---------- Solver 选项 ----------
        so = ocp.solver_options
        so.qp_solver       = "PARTIAL_CONDENSING_HPIPM"
        so.hessian_approx  = "GAUSS_NEWTON"
        so.integrator_type = "ERK"
        so.print_level     = 0
        so.nlp_solver_type = "SQP_RTI"

        # ---------- 生成 Solver ----------
        json_file = f"./{m_model.name}_acados_ocp.json"
        self.solver     = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)
