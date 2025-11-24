"""
MAGPIE ADCS v0.3 – Tables + optional figures

Features:
- Orbiter: RW250 3-axis cluster
- Probe:   RW25  3-axis cluster

Data sets:
1) Orbiter slews about X, Y, Z:
   angles = [5, 15, 30, 60, 90, 180] deg
   -> per-case printouts + combined summary table.

2) Probe slews about +Y:
   angles = [1, 5, 10, 20, 30] deg
   -> per-case printouts + summary table.

3) Orbiter nadir-hold with gravity-gradient:
   altitudes = [250, 500, 1000] km
   -> per-altitude:
      orbital period, RW energy/orbit, RW energy/sol,
      avg & peak RW power.

Figures (controlled by toggles below):
- Orbiter 60° slew about +Y (multi-panel).
- Nadir-hold energy per sol vs altitude.
- Nadir-hold average RW power vs altitude.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------
# Global figure toggles
# --------------------------------------------
SHOW_FIGURES = True       # show matplotlib windows
SAVE_FIGURES = True       # save PNGs to FIG_DIR
FIG_DIR       = "Figures" # directory for outputs


def maybe_save_show(fig, filename_base):
    """Save and/or show figure depending on global toggles."""
    if SAVE_FIGURES:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, filename_base)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[FIG] Saved {path}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------
# Quaternion + rotation helpers
# --------------------------------------------

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    s = np.sin(angle_rad / 2.0)
    return np.array([np.cos(angle_rad / 2.0), *(axis * s)])


def omega_matrix(omega):
    wx, wy, wz = omega
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0, wx],
        [wz,  wy, -wx, 0.0]
    ])


def attitude_error_vec(q, q_desired):
    """Small-angle attitude error vector in body frame."""
    q = quat_normalize(q)
    qd = quat_normalize(q_desired)
    q_conj = q.copy()
    q_conj[1:] *= -1.0
    q_err = quat_mul(qd, q_conj)
    return q_err[1:]  # vector part


def rotmat_from_quat(q):
    """Rotation matrix R_bi from body to inertial."""
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]
    ])


def quat_from_dcm(R):
    """Rotation matrix -> quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif (R[1,1] > R[2,2]):
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
    return quat_normalize(np.array([w, x, y, z]))


# --------------------------------------------
# Reaction wheel cluster + rigid body
# --------------------------------------------

class ReactionWheelCluster:
    """
    N reaction wheels with fixed spin axes in body frame.

    axes: 3xN matrix, each column a unit axis
    Iw: rotor inertia [kg·m²]
    tau_max: max wheel torque magnitude [N·m]
    w_max:   max wheel speed magnitude [rad/s]
    """
    def __init__(self, axes, Iw, tau_max, w_max):
        self.A = np.array(axes, dtype=float)  # 3 x N
        for i in range(self.A.shape[1]):
            n = np.linalg.norm(self.A[:, i])
            if n > 0:
                self.A[:, i] /= n
        self.Iw = float(Iw)
        self.tau_max = float(tau_max)
        self.w_max = float(w_max)
        self.A_pinv = np.linalg.pinv(self.A)

    def allocate(self, tau_body_cmd):
        tau_body_cmd = np.asarray(tau_body_cmd, dtype=float)
        # body torque = -A * tau_rw  => tau_rw = -A^+ * tau_body_cmd
        return -self.A_pinv @ tau_body_cmd

    def apply_limits(self, tau_rw, w_rw):
        tau_rw = np.clip(tau_rw, -self.tau_max, self.tau_max)
        for i in range(len(w_rw)):
            if w_rw[i] >= self.w_max and tau_rw[i] > 0:
                tau_rw[i] = 0.0
            if w_rw[i] <= -self.w_max and tau_rw[i] < 0:
                tau_rw[i] = 0.0
        return tau_rw

    def body_torque(self, tau_rw):
        return -self.A @ tau_rw

    def wdot(self, tau_rw):
        return tau_rw / self.Iw


class RigidBodyWithRWs:
    """
    State = [ q(4), w_body(3), w_rw(N) ]
    """
    def __init__(self, I_body, rw_cluster, name="vehicle"):
        self.I = np.array(I_body, dtype=float)
        self.I_inv = np.linalg.inv(self.I)
        self.rw = rw_cluster
        self.name = name

    def dynamics(self, t, state, tau_body_cmd_func, q_des):
        q = quat_normalize(state[0:4])
        w_body = state[4:7]
        w_rw = state[7:]

        tau_cmd = tau_body_cmd_func(t, q, w_body, q_des)

        tau_rw = self.rw.allocate(tau_cmd)
        tau_rw = self.rw.apply_limits(tau_rw, w_rw)

        tau_body = self.rw.body_torque(tau_rw)

        Iw = self.I @ w_body
        wdot = self.I_inv @ (tau_body - np.cross(w_body, Iw))
        qdot = 0.5 * omega_matrix(w_body) @ q
        w_rw_dot = self.rw.wdot(tau_rw)

        return np.concatenate((qdot, wdot, w_rw_dot))


# --------------------------------------------
# RK4 integrator
# --------------------------------------------

def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = f(t + dt, y + dt*k3)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4) / 6.0


def simulate(state0, dyn_func, t_final, dt):
    n = int(np.ceil(t_final / dt))
    t = 0.0
    state = state0.copy()

    hist_t = np.zeros(n + 1)
    hist_state = np.zeros((n + 1, len(state0)))
    hist_t[0] = t
    hist_state[0] = state

    for k in range(1, n + 1):
        state = rk4_step(dyn_func, t, state, dt)
        state[0:4] = quat_normalize(state[0:4])
        t += dt
        hist_t[k] = t
        hist_state[k] = state

    return {"t": hist_t, "state": hist_state}


# --------------------------------------------
# Utility: RW torque & power from history
# --------------------------------------------

def compute_rw_power(history, Iw, eta=0.7):
    """
    Estimate RW torque and power from wheel speeds:

    tau = Iw * d(omega_w)/dt
    P_el = |tau * omega_w| / eta
    """
    t = history["t"]
    state = history["state"]
    dt = t[1] - t[0]

    w_rw = state[:, 7:]          # [steps, Nw]

    # central difference for d(omega)/dt
    wdot = np.zeros_like(w_rw)
    wdot[1:-1] = (w_rw[2:] - w_rw[:-2]) / (2*dt)
    wdot[0] = (w_rw[1] - w_rw[0]) / dt
    wdot[-1] = (w_rw[-1] - w_rw[-2]) / dt

    tau_rw = Iw * wdot                      # [steps, Nw]
    P_mech = np.abs(tau_rw * w_rw)          # elementwise
    P_el = P_mech / eta

    P_el_tot = np.sum(P_el, axis=1)         # [steps]
    E_J = np.trapezoid(P_el_tot, t)         # Joules
    E_Wh = E_J / 3600.0

    return {
        "tau_rw": tau_rw,
        "P_el": P_el,
        "P_el_tot": P_el_tot,
        "E_J": E_J,
        "E_Wh": E_Wh
    }


def estimate_settling_time(history,
                           w_thresh_deg=0.01,
                           ang_err_thresh_deg=0.1):
    """
    Settling time:
    first time |w| < w_thresh AND minimal-angle attitude error < ang_thresh
    relative to final quaternion.
    """
    t = history["t"]
    state = history["state"]
    w_body = state[:, 4:7]
    q_hist = state[:, 0:4]

    q_final = q_hist[-1]
    w_mag_deg = np.rad2deg(np.linalg.norm(w_body, axis=1))

    # attitude error vs final (minimal angle)
    ang_err = np.zeros_like(t)
    for k in range(len(t)):
        q_conj_f = q_final.copy()
        q_conj_f[1:] *= -1.0
        q_delta = quat_mul(q_conj_f, q_hist[k])
        ang = 2 * np.arccos(np.clip(q_delta[0], -1.0, 1.0))
        # wrap to [0, pi]
        if ang > np.pi:
            ang = 2*np.pi - ang
        ang_err[k] = np.rad2deg(ang)

    for k in range(len(t)):
        if (w_mag_deg[k] < w_thresh_deg) and (ang_err[k] < ang_err_thresh_deg):
            return t[k]

    return t[-1]


# --------------------------------------------
# Plot helpers
# --------------------------------------------

def plot_slew_history(history, power, title="Slew case", filename="slew_case.png"):
    """Multi-panel plot for a single slew: body rates, att error, RW speeds, RW power."""
    t = history["t"]
    state = history["state"]
    w_body = state[:, 4:7]             # rad/s
    q_hist = state[:, 0:4]
    w_rw = state[:, 7:]                # rad/s
    P_tot = power["P_el_tot"]          # W

    # Body rates [deg/s]
    w_deg = np.rad2deg(w_body)

    # Attitude error vs final attitude (minimal angle)
    q_final = q_hist[-1]
    ang_err_deg = np.zeros_like(t)
    for k in range(len(t)):
        q_conj_f = q_final.copy()
        q_conj_f[1:] *= -1.0
        q_delta = quat_mul(q_conj_f, q_hist[k])
        ang = 2 * np.arccos(np.clip(q_delta[0], -1.0, 1.0))
        if ang > np.pi:
            ang = 2*np.pi - ang
        ang_err_deg[k] = np.rad2deg(ang)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title)

    # Body rates
    ax = axes[0]
    ax.plot(t, w_deg[:, 0], label="ωx")
    ax.plot(t, w_deg[:, 1], label="ωy")
    ax.plot(t, w_deg[:, 2], label="ωz")
    ax.set_ylabel("Body rates [deg/s]")
    ax.grid(True)
    ax.legend()

    # Attitude error
    ax = axes[1]
    ax.plot(t, ang_err_deg)
    ax.set_ylabel("Attitude error [deg]")
    ax.grid(True)

    # RW speeds
    ax = axes[2]
    for i in range(w_rw.shape[1]):
        ax.plot(t, w_rw[:, i], label=f"RW{i+1}")
    ax.set_ylabel("RW speed [rad/s]")
    ax.grid(True)
    ax.legend()

    # RW total power
    ax = axes[3]
    ax.plot(t, P_tot)
    ax.set_ylabel("P_RW,total [W]")
    ax.set_xlabel("Time [s]")
    ax.grid(True)

    plt.tight_layout()
    maybe_save_show(fig, filename)


# --------------------------------------------
# Constants
# --------------------------------------------
DEG2RAD = np.pi / 180.0

# ASTROFEIN wheel numbers (approx)
# RW250
IW_RW250       = 7.65e-3                  # kg·m²
TAU_MAX_RW250  = 0.10                     # N·m
W_MAX_RW250    = 5000 * 2*np.pi/60.0      # rad/s
ETA_RW250      = 0.7                      # efficiency guess

# RW25 (~0.03 Nms at 5000 rpm)
H_RW25         = 0.03                     # N·m·s
W_NOM_RW25     = 5000 * 2*np.pi/60.0
IW_RW25        = H_RW25 / W_NOM_RW25      # kg·m²
TAU_MAX_RW25   = 0.002                    # N·m
W_MAX_RW25     = 5500 * 2*np.pi/60.0      # rad/s
ETA_RW25       = 0.6                      # efficiency guess

# Mars gravity
MU_MARS  = 4.282837e13    # m^3/s^2
R_MARS   = 3389.5e3       # m


# --------------------------------------------
# Gravity-gradient torque for Mars orbit
# --------------------------------------------

def gravity_gradient_torque(I_body, q_body_to_inertial, r_mag, theta):
    """
    Simple equatorial circular orbit:
        r_hat_inertial = [cos(theta), sin(theta), 0]
    theta = n * t
    """
    r_hat_inertial = np.array([np.cos(theta), np.sin(theta), 0.0])
    R_bi = rotmat_from_quat(q_body_to_inertial)
    r_hat_body = R_bi @ r_hat_inertial

    factor = 3 * MU_MARS / (r_mag**3)
    return factor * np.cross(r_hat_body, I_body @ r_hat_body)


# --------------------------------------------
# 1) Orbiter slew tables about X, Y, Z
# --------------------------------------------

def run_orbiter_slew_tables_all_axes():
    # Orbiter inertia – placeholder; replace with CAD if desired
    I_orbiter = np.diag([250.0, 220.0, 300.0])

    # 3 wheels along body axes
    axes_rw = np.eye(3)
    rw_cluster = ReactionWheelCluster(
        axes=axes_rw,
        Iw=IW_RW250,
        tau_max=TAU_MAX_RW250,
        w_max=W_MAX_RW250
    )

    Kp = 0.2
    Kd = 25.0

    def tau_body_cmd(t, q, w_body, q_desired):
        e = attitude_error_vec(q, q_desired)
        return -Kp * e - Kd * w_body

    q_des = np.array([1.0, 0.0, 0.0, 0.0])

    slew_angles_deg = [5, 15, 30, 60, 90, 180]
    axis_labels = ["X", "Y", "Z"]
    axis_vectors = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }

    all_results = {ax: [] for ax in axis_labels}

    print("\n=== Orbiter Slew Cases (RW250, about body X/Y/Z) ===")

    for ax_label in axis_labels:
        axis_vec = axis_vectors[ax_label]
        print(f"\n>>> Axis: {ax_label}")
        vehicle = RigidBodyWithRWs(I_orbiter, rw_cluster, name=f"Orbiter_{ax_label}")

        for ang_deg in slew_angles_deg:
            ang_rad = ang_deg * DEG2RAD
            q0 = quat_from_axis_angle(axis_vec, ang_rad)
            w0 = np.zeros(3)
            w_rw0 = np.zeros(3)
            state0 = np.concatenate((q0, w0, w_rw0))

            t_final = 2000.0
            dt = 0.5

            def dyn(t, state):
                return vehicle.dynamics(t, state, tau_body_cmd, q_des)

            hist = simulate(state0, dyn_func=dyn, t_final=t_final, dt=dt)
            power = compute_rw_power(hist, Iw=IW_RW250, eta=ETA_RW250)

            w_body = hist["state"][:, 4:7]
            w_mag_deg = np.rad2deg(np.linalg.norm(w_body, axis=1))
            max_slew = np.max(w_mag_deg)
            t_settle = estimate_settling_time(hist)

            E_J = power["E_J"]
            E_Wh = power["E_Wh"]
            P_avg = E_J / hist["t"][-1]
            P_peak = np.max(power["P_el_tot"])

            all_results[ax_label].append({
                "angle_deg": ang_deg,
                "t_settle": t_settle,
                "max_slew": max_slew,
                "E_J": E_J,
                "E_Wh": E_Wh,
                "P_avg": P_avg,
                "P_peak": P_peak
            })

            print(f"  Slew {ang_deg:6.1f} deg:"
                  f"  t_settle={t_settle:7.1f} s,"
                  f"  |w|max={max_slew:6.3f} deg/s,"
                  f"  E={E_J:7.1f} J,"
                  f"  Pavg={P_avg:5.3f} W,"
                  f"  Ppeak={P_peak:6.3f} W")

    # Combined summary tables
    for ax_label in axis_labels:
        results = all_results[ax_label]
        print(f"\n=== Slew Summary Table (Orbiter + RW250, axis {ax_label}) ===")
        header = ("Angle[deg]", "t_settle[s]", "max_slew[deg/s]",
                  "E[J]", "E[Wh]", "Pavg[W]", "Ppeak[W]")
        print("{:>10s} {:>12s} {:>16s} {:>12s} {:>12s} {:>10s} {:>10s}".format(*header))
        for r in results:
            print("{:10.1f} {:12.1f} {:16.3f} {:12.2f} {:12.5f} {:10.3f} {:10.3f}".format(
                r["angle_deg"], r["t_settle"], r["max_slew"],
                r["E_J"], r["E_Wh"], r["P_avg"], r["P_peak"]
            ))


# --------------------------------------------
# Extra: example orbiter slew plot (60° about +Y)
# --------------------------------------------

def run_example_orbiter_slew_plot():
    """Re-run a single 60° +Y slew and make detailed plots."""
    I_orbiter = np.diag([250.0, 220.0, 300.0])

    axes_rw = np.eye(3)
    rw_cluster = ReactionWheelCluster(
        axes=axes_rw,
        Iw=IW_RW250,
        tau_max=TAU_MAX_RW250,
        w_max=W_MAX_RW250
    )

    Kp = 0.2
    Kd = 25.0

    def tau_body_cmd(t, q, w_body, q_desired):
        e = attitude_error_vec(q, q_desired)
        return -Kp * e - Kd * w_body

    q_des = np.array([1.0, 0.0, 0.0, 0.0])

    axis_vec = np.array([0.0, 1.0, 0.0])  # +Y
    ang_deg = 60.0
    ang_rad = ang_deg * DEG2RAD

    q0 = quat_from_axis_angle(axis_vec, ang_rad)
    w0 = np.zeros(3)
    w_rw0 = np.zeros(3)
    state0 = np.concatenate((q0, w0, w_rw0))

    t_final = 2000.0
    dt = 0.5

    vehicle = RigidBodyWithRWs(I_orbiter, rw_cluster, name="Orbiter_Y_60deg")

    def dyn(t, state):
        return vehicle.dynamics(t, state, tau_body_cmd, q_des)

    hist = simulate(state0, dyn_func=dyn, t_final=t_final, dt=dt)
    power = compute_rw_power(hist, Iw=IW_RW250, eta=ETA_RW250)

    plot_slew_history(
        hist,
        power,
        title="Orbiter 60° slew about +Y",
        filename="adcs_slew_60deg_Y.png"
    )


# --------------------------------------------
# 2) Probe slews with RW25 cluster
# --------------------------------------------

def run_probe_slew_table():
    # Probe + aeroshell inertia – replace with exact CAD if available
    Ixx_gmm2 = 3.9596118029e10
    Iyy_gmm2 = 3.9596118029e10
    Izz_gmm2 = 5.652922409e9
    I_probe = np.diag([
        Ixx_gmm2 * 1e-9,
        Iyy_gmm2 * 1e-9,
        Izz_gmm2 * 1e-9
    ])

    axes_rw = np.eye(3)
    rw_cluster = ReactionWheelCluster(
        axes=axes_rw,
        Iw=IW_RW25,
        tau_max=TAU_MAX_RW25,
        w_max=W_MAX_RW25
    )
    vehicle = RigidBodyWithRWs(I_probe, rw_cluster, name="MAGPIE_Probe")

    q_des = np.array([1.0, 0.0, 0.0, 0.0])

    # smaller gains for tiny RW25
    Kp = 0.1
    Kd = 1.5

    def tau_body_cmd(t, q, w_body, q_desired):
        e = attitude_error_vec(q, q_desired)
        return -Kp * e - Kd * w_body

    slew_angles_deg = [1, 5, 10, 20, 30]
    axis_vec = np.array([0.0, 1.0, 0.0])  # about +Y

    print("\n=== Probe Slew Cases (RW25, about +Y) ===")

    results = []

    for ang_deg in slew_angles_deg:
        ang_rad = ang_deg * DEG2RAD
        q0 = quat_from_axis_angle(axis_vec, ang_rad)
        w0 = np.zeros(3)
        w_rw0 = np.zeros(3)
        state0 = np.concatenate((q0, w0, w_rw0))

        t_final = 2000.0
        dt = 0.5

        def dyn(t, state):
            return vehicle.dynamics(t, state, tau_body_cmd, q_des)

        hist = simulate(state0, dyn_func=dyn, t_final=t_final, dt=dt)
        power = compute_rw_power(hist, Iw=IW_RW25, eta=ETA_RW25)

        w_body = hist["state"][:, 4:7]
        w_mag_deg = np.rad2deg(np.linalg.norm(w_body, axis=1))
        max_slew = np.max(w_mag_deg)
        t_settle = estimate_settling_time(hist)

        E_J = power["E_J"]
        E_Wh = power["E_Wh"]
        P_avg = E_J / hist["t"][-1]
        P_peak = np.max(power["P_el_tot"])

        results.append({
            "angle_deg": ang_deg,
            "t_settle": t_settle,
            "max_slew": max_slew,
            "E_J": E_J,
            "E_Wh": E_Wh,
            "P_avg": P_avg,
            "P_peak": P_peak
        })

        print(f"  Slew {ang_deg:6.1f} deg:"
              f"  t_settle={t_settle:7.1f} s,"
              f"  |w|max={max_slew:6.3f} deg/s,"
              f"  E={E_J:7.3f} J,"
              f"  Pavg={P_avg:7.4f} W,"
              f"  Ppeak={P_peak:7.4f} W")

    print("\n=== Slew Summary Table (Probe + RW25, axis Y) ===")
    header = ("Angle[deg]", "t_settle[s]", "max_slew[deg/s]",
              "E[J]", "E[Wh]", "Pavg[W]", "Ppeak[W]")
    print("{:>10s} {:>12s} {:>16s} {:>12s} {:>12s} {:>10s} {:>10s}".format(*header))
    for r in results:
        print("{:10.1f} {:12.1f} {:16.3f} {:12.3f} {:12.6f} {:10.4f} {:10.4f}".format(
            r["angle_deg"], r["t_settle"], r["max_slew"],
            r["E_J"], r["E_Wh"], r["P_avg"], r["P_peak"]
        ))


# --------------------------------------------
# 3) Orbiter nadir-hold vs altitude
# --------------------------------------------

def simulate_orbiter_nadir_hold_multi_altitudes():
    # Orbiter inertia – same as above
    I_orbiter = np.diag([250.0, 220.0, 300.0])

    axes_rw = np.eye(3)
    rw_cluster = ReactionWheelCluster(
        axes=axes_rw,
        Iw=IW_RW250,
        tau_max=TAU_MAX_RW250,
        w_max=W_MAX_RW250
    )

    # altitudes in km
    altitudes_km = [250.0, 500.0, 1000.0]

    print("\n=== Orbiter Nadir-Hold vs Altitude (Mars, circular orbit) ===")

    results = []

    for h_km in altitudes_km:
        h = h_km * 1e3
        r_orbit = R_MARS + h
        n = np.sqrt(MU_MARS / r_orbit**3)   # mean motion [rad/s]
        T_orbit = 2 * np.pi / n

        Kp = 0.3
        Kd = 30.0

        def q_des_nadir(t):
            """
            LVLH / nadir pointing:
            - r_hat = [cos(nt), sin(nt), 0]
            - t_hat (along-track) = [-sin(nt), cos(nt), 0]
            - h_hat (orbit normal) = [0, 0, 1]
            body +X -> along-track
            body +Y -> orbit normal
            body -Z -> nadir
            """
            theta = n * t
            r_hat = np.array([np.cos(theta), np.sin(theta), 0.0])
            t_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])
            h_hat = np.array([0.0, 0.0, 1.0])

            t_hat /= np.linalg.norm(t_hat)
            r_hat /= np.linalg.norm(r_hat)
            h_hat /= np.linalg.norm(h_hat)

            ex_b = t_hat
            ey_b = h_hat
            ez_b = -r_hat

            R_bi = np.column_stack((ex_b, ey_b, ez_b))
            return quat_from_dcm(R_bi)

        def dyn(t, state):
            q = quat_normalize(state[0:4])
            w_body = state[4:7]
            w_rw = state[7:]

            q_des = q_des_nadir(t)
            e = attitude_error_vec(q, q_des)
            tau_cmd = -Kp * e - Kd * w_body

            tau_rw = rw_cluster.allocate(tau_cmd)
            tau_rw = rw_cluster.apply_limits(tau_rw, w_rw)

            tau_rw_body = rw_cluster.body_torque(tau_rw)

            theta = n * t
            tau_gg = gravity_gradient_torque(I_orbiter, q, r_mag=r_orbit, theta=theta)

            tau_body = tau_rw_body + tau_gg

            Iw = I_orbiter @ w_body
            wdot = np.linalg.inv(I_orbiter) @ (tau_body - np.cross(w_body, Iw))
            qdot = 0.5 * omega_matrix(w_body) @ q
            w_rw_dot = rw_cluster.wdot(tau_rw)

            return np.concatenate((qdot, wdot, w_rw_dot))

        q0 = q_des_nadir(0.0)
        w0 = np.zeros(3)
        w_rw0 = np.zeros(3)
        state0 = np.concatenate((q0, w0, w_rw0))

        dt = 1.0
        hist = simulate(state0, dyn_func=dyn, t_final=T_orbit, dt=dt)
        power = compute_rw_power(hist, Iw=IW_RW250, eta=ETA_RW250)

        w_body = hist["state"][:, 4:7]
        w_mag_deg = np.rad2deg(np.linalg.norm(w_body, axis=1))
        max_slew = np.max(w_mag_deg)

        E_J = power["E_J"]
        E_Wh = power["E_Wh"]
        P_avg = E_J / hist["t"][-1]
        P_peak = np.max(power["P_el_tot"])

        sol_hours = 24.6
        orbits_per_sol = sol_hours / (T_orbit / 3600.0)
        E_Wh_per_sol = E_Wh * orbits_per_sol

        results.append({
            "h_km": h_km,
            "T_orbit_min": T_orbit / 60.0,
            "max_slew_deg_s": max_slew,
            "E_Wh_orbit": E_Wh,
            "E_Wh_sol": E_Wh_per_sol,
            "P_avg": P_avg,
            "P_peak": P_peak
        })

        print(f"\n  Altitude: {h_km:6.1f} km")
        print(f"    Orbital period: {T_orbit/60.0:7.2f} min")
        print(f"    Max |w|: {max_slew:7.4f} deg/s")
        print(f"    RW energy / orbit: {E_Wh:7.4f} Wh")
        print(f"    RW energy / sol:   {E_Wh_per_sol:7.2f} Wh")
        print(f"    Avg RW power:      {P_avg:7.4f} W")
        print(f"    Peak RW power:     {P_peak:7.2f} W")

    # Summary table
    print("\n=== Nadir-Hold Summary Table (Orbiter + RW250 vs Altitude) ===")
    header = ("h[km]", "T_orbit[min]", "max|w|[deg/s]",
              "E_orbit[Wh]", "E_sol[Wh]", "Pavg[W]", "Ppeak[W]")
    print("{:>8s} {:>14s} {:>15s} {:>14s} {:>12s} {:>10s} {:>10s}".format(*header))
    for r in results:
        print("{:8.1f} {:14.2f} {:15.4f} {:14.4f} {:12.2f} {:10.4f} {:10.2f}".format(
            r["h_km"], r["T_orbit_min"], r["max_slew_deg_s"],
            r["E_Wh_orbit"], r["E_Wh_sol"], r["P_avg"], r["P_peak"]
        ))

    # Plots: energy per sol and average power vs altitude
    h_vals = np.array([r["h_km"] for r in results])
    E_sol_vals = np.array([r["E_Wh_sol"] for r in results])
    P_avg_vals = np.array([r["P_avg"] for r in results])

    fig1 = plt.figure(figsize=(6, 4))
    ax1 = fig1.add_subplot(111)
    ax1.plot(h_vals, E_sol_vals, marker="o")
    ax1.set_xlabel("Altitude [km]")
    ax1.set_ylabel("RW energy per sol [Wh]")
    ax1.set_title("Nadir-hold RW energy vs altitude (Mars)")
    ax1.grid(True)
    maybe_save_show(fig1, "adcs_nadir_energy_vs_alt.png")

    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(111)
    ax2.plot(h_vals, P_avg_vals, marker="o")
    ax2.set_xlabel("Altitude [km]")
    ax2.set_ylabel("Average RW power [W]")
    ax2.set_title("Nadir-hold average RW power vs altitude")
    ax2.grid(True)
    maybe_save_show(fig2, "adcs_nadir_pavg_vs_alt.png")


# --------------------------------------------
# Main
# --------------------------------------------

if __name__ == "__main__":
    # 1) Orbiter slew data for body X/Y/Z
    run_orbiter_slew_tables_all_axes()

    # Extra: detailed plots for a representative 60° +Y slew
    run_example_orbiter_slew_plot()

    # 2) Probe slew data for RW25 cluster
    run_probe_slew_table()

    # 3) Orbiter nadir-hold vs altitude (and plots)
    simulate_orbiter_nadir_hold_multi_altitudes()
