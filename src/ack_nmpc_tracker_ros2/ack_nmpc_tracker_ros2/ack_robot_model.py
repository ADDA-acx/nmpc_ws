#!/usr/bin/env python
# coding: UTF-8
"""
Single-track bicycle model for NMPC demos
Author: Wei Luo (orig.) / 2025-07-16 edit by ChatGPT
"""

import numpy as np
import casadi as ca
from acados_template import AcadosModel


class MobileRobotModel(object):
    """Front-steer bicycle model (x, y, θ) + controls (v, δ)."""

    def __init__(self, wheelbase: float = 1.16):
        """wheelbase: 轴距 L (m)"""
        self.L = float(wheelbase)

        # ---------- 1. CasADi symbols ----------
        v      = ca.SX.sym("v")          # linear velocity (m/s)
        delta  = ca.SX.sym("delta")      # steering angle (rad)
        controls = ca.vertcat(v, delta)

        x      = ca.SX.sym("x")
        y      = ca.SX.sym("y")
        theta  = ca.SX.sym("theta")
        states = ca.vertcat(x, y, theta)

        # ---------- 2. Continuous-time RHS ----------
        rhs = ca.vertcat(
            v * ca.cos(theta),                       # ẋ
            v * ca.sin(theta),                       # ẏ
            (v / self.L) * ca.tan(delta)             # θ̇
        )

        # ---------- 3. Build Acados model ----------
        model = AcadosModel()
        x_dot = ca.SX.sym("x_dot", rhs.shape[0])

        model.f_expl_expr = rhs
        model.f_impl_expr = x_dot - rhs
        model.x           = states
        model.xdot        = x_dot
        model.u           = controls
        model.p           = []            # no time-varying params
        model.name        = "bicycle_robot"

        # ---------- 4. Hard constraints ----------
        constraint = ca.types.SimpleNamespace()
        constraint.v_max     = 0.6            # m/s   (adjust!)
        constraint.v_min     = -0.6          # m/s   (adjust!)
        constraint.delta_max = 0.5            # ≈ 34°
        constraint.delta_min = -0.5           # ≈ -34°
        constraint.expr      = ca.vertcat(v, delta)

        # ---------- 5. Export ----------
        self.model      = model
        self.constraint = constraint

        p_sym = ca.SX.sym('p', 2)      # p[0]=v_prev, p[1]=delta_prev
        model.p = p_sym                # 告诉 acados 这是 stage-wise 参数
