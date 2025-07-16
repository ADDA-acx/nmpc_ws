#!/usr/bin/env python
# coding=UTF-8
"""
Author: Wei Luo  (original demo)
Minor edits: 2025‑07‑01 – kept interface unchanged so that it can be reused for
point‑stabilisation or trajectory‑tracking MPC examples.
"""

import numpy as np
import casadi as ca
from acados_template import AcadosModel


class MobileRobotModel(object):
    """Unicycle‑like mobile robot model for NMPC demos."""

    def __init__(self):
        model = AcadosModel()
        constraint = ca.types.SimpleNamespace()

        # -------------------------- control inputs --------------------------
        v = ca.SX.sym("v")        # linear velocity
        omega = ca.SX.sym("omega")  # angular velocity
        controls = ca.vertcat(v, omega)

        # ----------------------------- states -------------------------------
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        theta = ca.SX.sym("theta")
        states = ca.vertcat(x, y, theta)

        # ----------------------- continuous dynamics -----------------------
        rhs = ca.vertcat(v * ca.cos(theta),
                         v * ca.sin(theta),
                         omega)

        # build CasADi function
        f = ca.Function("f", [states, controls], [rhs],
                        ["state", "control_input"], ["rhs"])

        x_dot = ca.SX.sym("x_dot", rhs.shape[0])
        model.f_expl_expr = rhs
        model.f_impl_expr = x_dot - rhs
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = []
        model.name = "mobile_robot"

        # ------------------------ hard constraints -------------------------
        constraint.v_max = 0.5
        constraint.v_min = -0.5
        constraint.omega_max = np.pi/4
        constraint.omega_min = -np.pi/4
        constraint.expr = ca.vcat([v, omega])  # for possible future use

        self.model = model
        self.constraint = constraint
