#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mobile_robot_opt.py – acados 0.5.x   NMPC optimizer
NONLINEAR_LS cost   (x, y, cosθ, sinθ, v, ω)
Tested on Ubuntu-22.04 + Python-3.10 + acados-0.5.0
"""

import os, sys, shutil
from pathlib import Path
import numpy as np
import scipy.linalg
import casadi as ca

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from .diff_robot_model import MobileRobotModel


def safe_mkdir_recursive(directory, overwrite=False):
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    elif overwrite:
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)


class MobileRobotOptimizer:
    def __init__(self, m_model, m_constraint, t_horizon: float, n_nodes: int):

        # ---------- 时域 & 维度 ----------
        self.T, self.N = float(t_horizon), int(n_nodes)
        self.nx = m_model.x.shape[0]
        self.nu = m_model.u.shape[0]

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
        ocp.dims.np = 0

        # ---------- COST ----------
        ocp.cost.cost_type   = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.cost_type_0 = "NONLINEAR_LS"      # ★ 必填，否则 *_0 被无视

        # y = [x, y, cosθ, sinθ, v, ω]
        x_sym, y_sym, th_sym = m_model.x[0], m_model.x[1], m_model.x[2]
        v_sym, om_sym        = m_model.u[0], m_model.u[1]

        expr_y   = ca.vertcat(x_sym, y_sym, ca.cos(th_sym),
                              ca.sin(th_sym), v_sym, om_sym)
        expr_y_e = ca.vertcat(x_sym, y_sym,
                              ca.cos(th_sym), ca.sin(th_sym))

        # —— 将残差写进 **model** 对象（0.5 API）
        ocp.model.cost_y_expr    = expr_y
        ocp.model.cost_y_expr_e  = expr_y_e
        ocp.model.cost_y_expr_0  = expr_y

        # 权重矩阵
        Q_pos = np.diag([5.0, 5.0])
        Q_ang = np.diag([3.0, 3.0])
        R     = np.diag([0.5, 2.0])

        W   = scipy.linalg.block_diag(Q_pos, Q_ang, R)   # 6×6
        W_e = scipy.linalg.block_diag(Q_pos, Q_ang)      # 4×4

        # —— 路径阶段
        ocp.cost.W   = W
        ocp.dims.ny  = 6
        ocp.cost.yref = np.zeros(6)

        # —— 终端阶段
        ocp.cost.W_e   = W_e
        ocp.dims.ny_e  = 4
        ocp.cost.yref_e = np.zeros(4)

        # —— 初始阶段
        ocp.cost.W_0   = W
        ocp.dims.ny_0  = 6
        ocp.cost.yref_0 = np.zeros(6)

        # ---------- 控制约束 ----------
        ocp.constraints.x0  = np.zeros(self.nx)
        ocp.constraints.lbu = np.array([m_constraint.v_min,
                                        m_constraint.omega_min])
        ocp.constraints.ubu = np.array([m_constraint.v_max,
                                        m_constraint.omega_max])
        ocp.constraints.idxbu = np.array([0, 1])

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
