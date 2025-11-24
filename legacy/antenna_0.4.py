# ============================================================
# MAGPIE Antenna Designer & Modeler 
# now with nec support + gain summary table
#
# Liam Piper
# 2025
# ============================================================

from __future__ import annotations
import os, csv, math
from dataclasses import dataclass
from typing import Dict, Any, List, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ===================== CONFIG =====================

F_HZ_DEFAULT          = 433_000_000    # ~433 MHz

WIRE_DIAM_MM_DEFAULT  = 1.6

GROUND_PLANE_RADIUS_M = 0.5
RADIALS_COUNT         = 4

LOOP_DIAM_M_DEFAULT   = 1.0
LOOP_TURNS_DEFAULT    = 1

HELIX_HEIGHT_M        = 1.2
HELIX_DIAM_M          = 0.18
HELIX_TURNS           = 8

AX_TURNS_DEFAULT      = 8
AX_SPACING_LAMBDA     = 0.23
AX_DIAM_LAMBDA        = 0.32

YAGI_DIR_COUNT        = 3
YAGI_BOOM_SPACING_L   = 0.2
YAGI_WIRE_DIAM_MM     = 1.6

DISH_DIAM_M           = 0.6
DISH_EFFICIENCY       = 0.55

SAVE_DIR              = "./antenna_outputs"
DO_SAVE_PLOTS         = True
DO_SHOW_PLOTS         = True
Z0_REF                = 50.0

EXPORT_NEC            = True
NEC_DIR               = os.path.join(SAVE_DIR, "nec")
NEC_FREQ_POINTS       = 101
NEC_FREQ_SPAN_FRAC    = 0.05   # ±5%

ENFORCE_SIZE_CONSTRAINT = True
SIZE_LIMIT_X_M = 0.09
SIZE_LIMIT_Y_M = 0.09
SIZE_LIMIT_Z_M = 0.09

# ===================== CONSTANTS =====================
MU0    = 4*math.pi*1e-7
EPS0   = 8.854187817e-12
C0     = 299_792_458.0
RHO_CU = 1.724e-8

# ===================== UTILS =====================
def wavelength(f_hz: float) -> float:
    return C0 / f_hz

def skin_depth(f_hz: float, rho: float = RHO_CU, mu_r: float = 1.0) -> float:
    return math.sqrt(rho / (math.pi * f_hz * MU0 * mu_r))

def wire_ac_resistance(length_m: float, f_hz: float, dia_m: float) -> float:
    delta = skin_depth(f_hz)
    a = dia_m/2
    if delta <= 0 or a <= 0:
        return 0.0
    return RHO_CU * length_m / (2*math.pi*a*delta)

def ensure_dir(p: str):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def compute_scale_factor(size_x: float, size_y: float, size_z: float) -> float:
    factors = [1.0]
    if size_x > 0:
        factors.append(SIZE_LIMIT_X_M / size_x)
    if size_y > 0:
        factors.append(SIZE_LIMIT_Y_M / size_y)
    if size_z > 0:
        factors.append(SIZE_LIMIT_Z_M / size_z)
    scale = min(factors)
    return scale if scale < 1.0 else 1.0

@dataclass
class Result:
    kind: str
    f_hz: float
    build: Dict[str, Any]
    estimates: Dict[str, float]
    notes: str
    sweep_freqs: np.ndarray
    sweep_Zin: np.ndarray
    bbox_xyz_m: Tuple[float,float,float]
    size_violation: bool

def check_size_constraint(size_x, size_y, size_z) -> bool:
    return (size_x > SIZE_LIMIT_X_M or
            size_y > SIZE_LIMIT_Y_M or
            size_z > SIZE_LIMIT_Z_M)

# ===================== DESIGNERS =====================
def design_dipole(f_hz: float, wire_d_mm: float) -> Result:
    lam = wavelength(f_hz)
    total_len = 0.95 * lam/2
    arm_len   = total_len/2

    size_x = total_len
    size_y = 0.01
    size_z = 0.01

    scale_factor = 1.0
    auto_scaled = False
    if ENFORCE_SIZE_CONSTRAINT:
        sf = compute_scale_factor(size_x, size_y, size_z)
        if sf < 1.0:
            auto_scaled = True
            scale_factor = sf
            total_len *= sf
            arm_len = total_len/2
            size_x *= sf
            size_y *= sf
            size_z *= sf

    Rin = 73.0
    Q = 11.0
    bw_frac = 1/Q

    rac = wire_ac_resistance(total_len, f_hz, wire_d_mm/1000)
    eff = Rin / (Rin + rac) if (Rin+rac) > 0 else 1.0
    gain_dbi = 2.15

    sweep_f = np.linspace(0.95*f_hz, 1.05*f_hz, 101)
    kx = Rin * Q / f_hz * 0.02
    Zin_sweep = Rin + 1j * kx*(sweep_f - f_hz)

    viol = check_size_constraint(size_x, size_y, size_z)

    build = {
        "total_length_m": total_len,
        "arm_length_m": arm_len,
        "wire_d_mm": wire_d_mm,
        "bbox_x_m": size_x,
        "bbox_y_m": size_y,
        "bbox_z_m": size_z,
        "size_scale_factor": scale_factor,
        "auto_scaled": auto_scaled,
    }
    estimates = {
        "Rin_ohm": Rin,
        "Q": Q,
        "BW_frac": bw_frac,
        "efficiency": eff,
        "gain_dBi": gain_dbi,
        "wire_Rac_ohm": rac,
    }
    notes = "Half-wave thin-wire dipole in free-space; height detunes in real life."
    if auto_scaled:
        notes += (
            f" Auto-rescaled by factor {scale_factor:.3f} to fit "
            f"{SIZE_LIMIT_X_M}x{SIZE_LIMIT_Y_M}x{SIZE_LIMIT_Z_M} m envelope."
        )

    return Result("dipole", f_hz, build, estimates, notes,
                  sweep_f, Zin_sweep, (size_x,size_y,size_z), viol)

def design_monopole(f_hz: float, wire_d_mm: float, gp_radius_m: float, radials: int) -> Result:
    lam = wavelength(f_hz)
    h = 0.237 * lam

    size_x = 2*gp_radius_m
    size_y = 2*gp_radius_m
    size_z = h

    scale_factor = 1.0
    auto_scaled = False
    gp_radius_eff = gp_radius_m
    if ENFORCE_SIZE_CONSTRAINT:
        sf = compute_scale_factor(size_x, size_y, size_z)
        if sf < 1.0:
            auto_scaled = True
            scale_factor = sf
            h *= sf
            gp_radius_eff *= sf
            size_x *= sf
            size_y *= sf
            size_z *= sf

    Rin = 36.5
    Rg = 10.0 / max(radials,1) * (0.25 / max(gp_radius_eff, 0.25))
    Rin_eff = Rin + Rg

    Q = 12.0
    bw_frac = 1/Q

    rac = wire_ac_resistance(h, f_hz, wire_d_mm/1000)
    eff = Rin_eff / (Rin_eff + rac) if (Rin_eff+rac) > 0 else 1.0
    gain_dbi = 5.16

    sweep_f = np.linspace(0.95*f_hz, 1.05*f_hz, 101)
    kx = Rin_eff * Q / f_hz * 0.02
    Zin_sweep = Rin_eff + 1j * kx*(sweep_f - f_hz)

    viol = check_size_constraint(size_x, size_y, size_z)

    build = {
        "height_m": h,
        "wire_d_mm": wire_d_mm,
        "gp_radius_m": gp_radius_eff,
        "radials": radials,
        "bbox_x_m": size_x,
        "bbox_y_m": size_y,
        "bbox_z_m": size_z,
        "size_scale_factor": scale_factor,
        "auto_scaled": auto_scaled,
    }
    estimates = {
        "Rin_ohm": Rin_eff,
        "Q": Q,
        "BW_frac": bw_frac,
        "efficiency": eff,
        "gain_dBi": gain_dbi,
        "wire_Rac_ohm": rac,
    }
    notes = "Quarter-wave vertical with radials."
    if auto_scaled:
        notes += (
            f" Auto-rescaled by factor {scale_factor:.3f} to fit "
            f"{SIZE_LIMIT_X_M}x{SIZE_LIMIT_Y_M}x{SIZE_LIMIT_Z_M} m envelope."
        )

    return Result("monopole", f_hz, build, estimates, notes,
                  sweep_f, Zin_sweep, (size_x,size_y,size_z), viol)

def loop_inductance_single_turn(r_m: float, wire_radius_m: float) -> float:
    return MU0 * r_m * (math.log(8*r_m/wire_radius_m) - 2)

def design_loop(f_hz: float, wire_d_mm: float, diam_m: float, turns: int) -> Result:
    r = diam_m/2
    size_x = diam_m
    size_y = diam_m
    size_z = wire_d_mm/1000.0

    scale_factor = 1.0
    auto_scaled = False
    if ENFORCE_SIZE_CONSTRAINT:
        sf = compute_scale_factor(size_x, size_y, size_z)
        if sf < 1.0:
            auto_scaled = True
            scale_factor = sf
            diam_m *= sf
            r = diam_m/2
            size_x *= sf
            size_y *= sf
            size_z *= sf

    a = wire_d_mm/2000
    A = math.pi * r*r
    lam = wavelength(f_hz)

    Rr = 31200.0 * (turns*A/(lam*lam))**2

    L = (turns**2) * loop_inductance_single_turn(r, a)
    C = 1.0 / ((2*math.pi*f_hz)**2 * L)

    wire_len = (2*math.pi*r) * turns
    Rac = wire_ac_resistance(wire_len, f_hz, wire_d_mm/1000)

    X = 2*math.pi*f_hz*L
    Rtot = Rr + Rac
    Q = X / max(Rtot, 1e-6)
    bw_frac = 1/max(Q,1e-6)
    eff = Rr / max(Rtot, 1e-9)

    sweep_f = np.linspace(0.95*f_hz, 1.05*f_hz, 101)
    Zin_sweep = []
    for ff in sweep_f:
        w = 2*math.pi*ff
        Xl = w*L
        Xc = -1/(w*C)
        Zin_sweep.append(Rtot + 1j*(Xl+Xc))
    Zin_sweep = np.array(Zin_sweep)

    viol = check_size_constraint(size_x, size_y, size_z)

    build = {
        "diam_m": diam_m,
        "turns": turns,
        "wire_d_mm": wire_d_mm,
        "tuning_cap_F": C,
        "inductance_H": L,
        "bbox_x_m": size_x,
        "bbox_y_m": size_y,
        "bbox_z_m": size_z,
        "size_scale_factor": scale_factor,
        "auto_scaled": auto_scaled,
    }
    estimates = {
        "Rr_ohm": Rr,
        "wire_Rac_ohm": Rac,
        "Q": Q,
        "BW_frac": bw_frac,
        "efficiency": eff,
        # no explicit gain estimate here: small loop ~ -1 to -3 dBi typically
    }
    notes = (
        "Small tuned loop. High circulating current, high voltage on cap."
    )
    if auto_scaled:
        notes += (
            f" Auto-rescaled by factor {scale_factor:.3f} to fit "
            f"{SIZE_LIMIT_X_M}x{SIZE_LIMIT_Y_M}x{SIZE_LIMIT_Z_M} m envelope."
        )

    return Result("loop", f_hz, build, estimates, notes,
                  sweep_f, Zin_sweep, (size_x,size_y,size_z), viol)

def design_nmode_helix(f_hz: float, wire_d_mm: float, height_m: float, form_diam_m: float, turns: int) -> Result:
    size_x = form_diam_m
    size_y = form_diam_m
    size_z = height_m

    scale_factor = 1.0
    auto_scaled = False
    if ENFORCE_SIZE_CONSTRAINT:
        sf = compute_scale_factor(size_x, size_y, size_z)
        if sf < 1.0:
            auto_scaled = True
            scale_factor = sf
            height_m *= sf
            form_diam_m *= sf
            size_x *= sf
            size_y *= sf
            size_z *= sf

    pitch = height_m / turns
    circumference = math.pi * form_diam_m
    wire_len = turns * math.sqrt(pitch**2 + circumference**2)

    eff_height = 0.92 * height_m
    lam = wavelength(f_hz)

    a = wire_d_mm/2000
    h = max(eff_height, 1e-3)
    Cg = 2*math.pi*EPS0*h / max(math.log(2*h/a), 1.1)
    Xc = -1.0/(2*math.pi*f_hz*Cg)

    A_area = (form_diam_m/2)**2 * math.pi
    Lh = MU0 * (turns**2) * A_area / max(height_m, 1e-6)
    Xl = 2*math.pi*f_hz*Lh

    Xtot = Xl + Xc
    L_add = 0.0
    if Xtot < 0:
        L_add = -Xtot / (2*math.pi*f_hz)

    Rin = 20.0 * (h/(lam/4))**2 * 36.5
    Rac = wire_ac_resistance(wire_len, f_hz, wire_d_mm/1000)
    Rtot = Rin + Rac
    eff = Rin / max(Rtot, 1e-9)

    Q = 200.0 * (lam/(2*math.pi*h))
    bw_frac = 1/max(Q,1e-6)
    gain_dbi = 1.5

    sweep_f = np.linspace(0.95*f_hz, 1.05*f_hz, 101)
    kx = (Xtot) / (1e-9 + (f_hz*0.01))
    Zin_sweep = Rtot + 1j * (Xtot + kx*(sweep_f - f_hz))

    viol = check_size_constraint(size_x, size_y, size_z)

    build = {
        "height_m": height_m,
        "form_diam_m": form_diam_m,
        "turns": turns,
        "wire_length_m": wire_len,
        "wire_d_mm": wire_d_mm,
        "helix_L_H": Lh,
        "required_series_L_H": max(L_add, 0.0),
        "bbox_x_m": size_x,
        "bbox_y_m": size_y,
        "bbox_z_m": size_z,
        "size_scale_factor": scale_factor,
        "auto_scaled": auto_scaled,
    }
    estimates = {
        "Rin_ohm": max(Rtot, 0.01),
        "efficiency": eff,
        "Q": Q,
        "BW_frac": bw_frac,
        "gain_dBi": gain_dbi,
        "X_total_ohm": Xtot,
    }
    notes = (
        "Normal-mode helix (short loaded vertical). Needs matching network."
    )
    if auto_scaled:
        notes += (
            f" Auto-rescaled by factor {scale_factor:.3f} to fit "
            f"{SIZE_LIMIT_X_M}x{SIZE_LIMIT_Y_M}x{SIZE_LIMIT_Z_M} m envelope."
        )

    return Result("nmode_helix", f_hz, build, estimates, notes,
                  sweep_f, Zin_sweep, (size_x,size_y,size_z), viol)

def design_axial_helix(f_hz: float, turns: int, spacing_l: float, diam_l: float) -> Result:
    lam = wavelength(f_hz)
    C = diam_l * lam
    D = C / math.pi
    S = spacing_l * lam
    L_total = turns * S

    size_x = D
    size_y = D
    size_z = L_total

    scale_factor = 1.0
    auto_scaled = False
    if ENFORCE_SIZE_CONSTRAINT:
        sf = compute_scale_factor(size_x, size_y, size_z)
        if sf < 1.0:
            auto_scaled = True
            scale_factor = sf
            D *= sf
            C = D * math.pi
            S *= sf
            L_total *= sf
            size_x *= sf
            size_y *= sf
            size_z *= sf

    g_lin = 15 * turns * (C/lam)**2 * (S/lam)
    gain_dbi = 10*math.log10(g_lin) if g_lin > 0 else -999.0
    beamwidth_deg = 52 * math.sqrt(lam**2 / max(C*S*turns, 1e-9))
    Zin = 140.0

    sweep_f = np.linspace(0.95*f_hz, 1.05*f_hz, 101)
    Rin = np.ones_like(sweep_f)*Zin
    Xin = 1j*(sweep_f - f_hz)*(Zin*0.02/f_hz)
    Zin_sweep = Rin + Xin

    viol = check_size_constraint(size_x, size_y, size_z)

    build = {
        "turns": turns,
        "diam_m": D,
        "circumference_m": C,
        "spacing_m": S,
        "boom_length_m": L_total,
        "bbox_x_m": size_x,
        "bbox_y_m": size_y,
        "bbox_z_m": size_z,
        "size_scale_factor": scale_factor,
        "auto_scaled": auto_scaled,
    }
    estimates = {
        "feed_R_ohm": Zin,
        "gain_dBi": gain_dbi,
        "beamwidth_deg": beamwidth_deg,
    }
    notes = (
        "Axial-mode helix (end-fire, circular polarization)."
    )
    if auto_scaled:
        notes += (
            f" Auto-rescaled by factor {scale_factor:.3f} to fit "
            f"{SIZE_LIMIT_X_M}x{SIZE_LIMIT_Y_M}x{SIZE_LIMIT_Z_M} m envelope."
        )

    return Result("axial_helix", f_hz, build, estimates, notes,
                  sweep_f, Zin_sweep, (size_x,size_y,size_z), viol)

def design_yagi(f_hz: float, dir_count: int, spacing_l: float, wire_d_mm: float) -> Result:
    lam = wavelength(f_hz)

    lengths: List[float] = []
    pos_x: List[float]   = []

    lengths.append(0.53*lam)
    pos_x.append(0.0)

    lengths.append(0.50*lam)
    pos_x.append(0.03*lam)

    for i in range(dir_count):
        lengths.append(0.47*lam * (0.98**i))
        pos_x.append(0.03*lam + (i+1)*spacing_l*lam)

    boom_len = max(pos_x) if pos_x else 0.0
    max_elem_len = max(lengths) if lengths else 0.0

    size_x = max_elem_len
    size_y = 0.05
    size_z = boom_len

    scale_factor = 1.0
    auto_scaled = False
    if ENFORCE_SIZE_CONSTRAINT:
        sf = compute_scale_factor(size_x, size_y, size_z)
        if sf < 1.0:
            auto_scaled = True
            scale_factor = sf
            lengths = [L*sf for L in lengths]
            pos_x   = [p*sf for p in pos_x]
            boom_len = max(pos_x) if pos_x else 0.0
            max_elem_len = max(lengths) if lengths else 0.0
            size_x *= sf
            size_y *= sf
            size_z *= sf

    gain_dbi = 7.0 + max(0, dir_count-1)*1.0
    fbr_dB   = 15.0
    Zin = 25.0

    sweep_f = np.linspace(0.95*f_hz, 1.05*f_hz, 101)
    Rin = np.ones_like(sweep_f)*Zin
    Xin = 1j*(sweep_f - f_hz)*(Zin*0.05/f_hz)
    Zin_sweep = Rin + Xin

    viol = check_size_constraint(size_x, size_y, size_z)

    build = {
        "elements_count": 2+dir_count,
        "element_lengths_m": lengths,
        "element_pos_along_boom_m": pos_x,
        "wire_d_mm": wire_d_mm,
        "boom_length_m": boom_len,
        "bbox_x_m": size_x,
        "bbox_y_m": size_y,
        "bbox_z_m": size_z,
        "size_scale_factor": scale_factor,
        "auto_scaled": auto_scaled,
    }
    estimates = {
        "approx_feed_R_ohm": Zin,
        "gain_dBi": gain_dbi,
        "front_to_back_dB": fbr_dB,
    }
    notes = "Quick Yagi-Uda layout guess."
    if auto_scaled:
        notes += (
            f" Auto-rescaled by factor {scale_factor:.3f} to fit "
            f"{SIZE_LIMIT_X_M}x{SIZE_LIMIT_Y_M}x{SIZE_LIMIT_Z_M} m envelope."
        )

    return Result("yagi", f_hz, build, estimates, notes,
                  sweep_f, Zin_sweep, (size_x,size_y,size_z), viol)

def design_dish(f_hz: float, D_m: float, eta: float) -> Result:
    depth_m = 0.25 * D_m
    size_x = D_m
    size_y = D_m
    size_z = depth_m

    scale_factor = 1.0
    auto_scaled = False
    if ENFORCE_SIZE_CONSTRAINT:
        sf = compute_scale_factor(size_x, size_y, size_z)
        if sf < 1.0:
            auto_scaled = True
            scale_factor = sf
            D_m *= sf
            depth_m = 0.25 * D_m
            size_x *= sf
            size_y *= sf
            size_z *= sf

    lam = wavelength(f_hz)

    gain_linear = eta * (math.pi * D_m / lam)**2
    gain_dbi = 10 * math.log10(gain_linear) if gain_linear > 0 else -999.0
    hpbw_deg = 70 * lam / max(D_m, 1e-9)

    Zin_center = 50.0
    sweep_f = np.linspace(0.95*f_hz, 1.05*f_hz, 101)
    Rin = np.ones_like(sweep_f) * Zin_center
    Xin = 1j*(sweep_f - f_hz)*(Zin_center*0.01/f_hz)
    Zin_sweep = Rin + Xin

    viol = check_size_constraint(size_x, size_y, size_z)

    build = {
        "dish_diam_m": D_m,
        "efficiency": eta,
        "est_beamwidth_deg": hpbw_deg,
        "bbox_x_m": size_x,
        "bbox_y_m": size_y,
        "bbox_z_m": size_z,
        "size_scale_factor": scale_factor,
        "auto_scaled": auto_scaled,
    }
    estimates = {
        "gain_dBi": gain_dbi,
        "feed_R_ohm": Zin_center,
    }
    notes = (
        "Parabolic dish with simple center feed. NEC export uses only a small feed dipole."
    )
    if auto_scaled:
        notes += (
            f" Auto-rescaled by factor {scale_factor:.3f} to fit "
            f"{SIZE_LIMIT_X_M}x{SIZE_LIMIT_Y_M}x{SIZE_LIMIT_Z_M} m envelope."
        )

    return Result("dish", f_hz, build, estimates, notes,
                  sweep_f, Zin_sweep, (size_x,size_y,size_z), viol)

# ===================== PATTERN MODELS =====================
def pat_dipole(theta):   return np.sin(theta)**2
def pat_monopole(theta):
    p = np.sin(theta)**2
    return np.where(theta > np.pi/2, 1e-6, p)
def pat_loop(theta):     return np.sin(theta)**2
def pat_nmode(theta):    return pat_monopole(theta)
def pat_axial(theta):    return 0.5*(1+np.cos(theta))**2
def pat_yagi(theta):
    fwd  = np.exp(-((theta-0.0)/0.3)**2)
    back = np.exp(-((theta-math.pi)/0.6)**2)*0.05
    return fwd + back
def pat_dish(theta):
    main = np.exp(-((theta-0.0)/0.12)**2)
    side = np.exp(-((theta-0.6)/0.1)**2)*0.02
    return main + side

# ===================== PLOTTING =====================
def gamma_from_Z(Z: complex, Z0=Z0_REF) -> complex:
    return (Z - Z0)/(Z + Z0)

def plot_polar_theta(pattern_fn, title, path):
    thetas = np.linspace(0, np.pi, 721)
    p = pattern_fn(thetas)
    p = np.maximum(p, 1e-6)
    p_db = 10*np.log10(p/np.max(p))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(thetas, p_db)
    ax.set_title(title)
    ax.set_rmin(-30)
    if path and DO_SAVE_PLOTS:
        fig.savefig(path, dpi=160, bbox_inches='tight')
    if DO_SHOW_PLOTS:
        plt.show()
    plt.close(fig)

def plot_3d_pattern(pattern_fn, title, path):
    thetas = np.linspace(0, np.pi, 181)
    phis   = np.linspace(0, 2*np.pi, 361)
    TH, PH = np.meshgrid(thetas, phis)

    P = pattern_fn(TH)
    P = np.maximum(P, 1e-9)
    Pn = P/np.max(P)

    R = Pn
    X = R*np.sin(TH)*np.cos(PH)
    Y = R*np.sin(TH)*np.sin(PH)
    Z = R*np.cos(TH)

    P_db = 10*np.log10(Pn)
    c_norm = (P_db - np.min(P_db))/(np.max(P_db)-np.min(P_db))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        X, Y, Z,
        facecolors=plt.cm.viridis(c_norm),
        linewidth=0,
        antialiased=False
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    max_range = np.max([np.ptp(X), np.ptp(Y), np.ptp(Z)]) / 2
    mid_x = (np.max(X)+np.min(X)) / 2
    mid_y = (np.max(Y)+np.min(Y)) / 2
    mid_z = (np.max(Z)+np.min(Z)) / 2

    ax.set_xlim(mid_x-max_range, mid_x+max_range)
    ax.set_ylim(mid_y-max_range, mid_y+max_range)
    ax.set_zlim(mid_z-max_range, mid_z+max_range)

    if path and DO_SAVE_PLOTS:
        fig.savefig(path, dpi=160, bbox_inches='tight')
    if DO_SHOW_PLOTS:
        plt.show()
    plt.close(fig)

def plot_smith(Z_sweep, title, path):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_title(title)

    phi = np.linspace(0,2*np.pi,400)
    ax.plot(np.cos(phi), np.sin(phi), linewidth=0.5)

    gammas = np.array([gamma_from_Z(z) for z in Z_sweep])
    ax.plot(gammas.real, gammas.imag, marker='o', markersize=2, linewidth=1)

    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_xlabel("Re(Gamma)")
    ax.set_ylabel("Im(Gamma)")

    if path and DO_SAVE_PLOTS:
        fig.savefig(path, dpi=160, bbox_inches='tight')
    if DO_SHOW_PLOTS:
        plt.show()
    plt.close(fig)

def plot_s11(f_sweep, Z_sweep, title, path):
    gammas = np.array([gamma_from_Z(z) for z in Z_sweep])
    s11_mag = np.abs(gammas)

    fig, ax = plt.subplots()
    ax.plot(f_sweep/1e6, 20*np.log10(s11_mag))
    ax.set_xlabel("Freq (MHz)")
    ax.set_ylabel("S11 (dB)")
    ax.set_title(title)
    ax.grid(True)

    if path and DO_SAVE_PLOTS:
        fig.savefig(path, dpi=160, bbox_inches='tight')
    if DO_SHOW_PLOTS:
        plt.show()
    plt.close(fig)

# ===================== OUTPUT =====================
def summarize(result: Result):
    print("\n=== Antenna Build Sheet ===")
    print(f"Kind: {result.kind}")
    print(f"Freq: {result.f_hz/1e6:.6f} MHz | lambda = {wavelength(result.f_hz):.4f} m")

    sx, sy, sz = result.bbox_xyz_m
    print(f"[Envelope] X:{sx:.3f} m  Y:{sy:.3f} m  Z:{sz:.3f} m")
    if result.size_violation:
        print(f"WARNING: exceeds limit ({SIZE_LIMIT_X_M} x {SIZE_LIMIT_Y_M} x {SIZE_LIMIT_Z_M} m)")

    for k,v in result.build.items():
        print(f"{k}: {v}")
    print("-- Estimates --")
    for k,v in result.estimates.items():
        if isinstance(v,(int,float)) and "ohm" in k.lower():
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")
    print(f"Notes: {result.notes}")

def write_csv_single(result: Result, outdir: str, basename: str):
    ensure_dir(outdir)
    path = os.path.join(outdir, f"{basename}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind","f_hz","param","value"])
        w.writerow(["info", f"{result.f_hz:.3f}",
                    "bbox_xyz_m",
                    f"{result.bbox_xyz_m} (violation={result.size_violation})"])
        for k,v in result.build.items():
            w.writerow([result.kind, f"{result.f_hz:.3f}", k, v])
        for k,v in result.estimates.items():
            w.writerow([result.kind, f"{result.f_hz:.3f}", k, v])
        w.writerow([result.kind, f"{result.f_hz:.3f}", "notes", result.notes])
    return path

def write_gain_summary_table(results: List[Result]):
    """
    Create a concise summary table of gains, efficiency, and size
    for all antennas. Printed to console and saved as CSV.
    """
    ensure_dir(SAVE_DIR)
    path = os.path.join(SAVE_DIR, "antenna_gain_summary.csv")

    # Console pretty-print
    print("\n=== Antenna Gain Summary (approx, free-space) ===")
    header = f"{'Kind':<15} {'f (MHz)':>8} {'Gain (dBi)':>12} {'Eff':>8} {'Bx×By×Bz (m)':>22} {'Size OK?':>9}"
    print(header)
    print("-" * len(header))

    rows: List[List[Any]] = []
    rows.append(["kind", "f_hz", "gain_dBi", "efficiency", "bbox_x_m", "bbox_y_m", "bbox_z_m", "size_violation"])

    for r in results:
        gx = r.estimates.get("gain_dBi", float("nan"))
        eff = r.estimates.get("efficiency", float("nan"))
        bx, by, bz = r.bbox_xyz_m
        size_ok = not r.size_violation

        # Console row
        gain_str = f"{gx:0.2f}" if np.isfinite(gx) else "n/a"
        eff_str  = f"{eff:0.2f}" if np.isfinite(eff) else "n/a"
        bbox_str = f"{bx:0.3f}×{by:0.3f}×{bz:0.3f}"
        print(f"{r.kind:<15} {r.f_hz/1e6:8.3f} {gain_str:>12} {eff_str:>8} {bbox_str:>22} {str(size_ok):>9}")

        rows.append([
            r.kind,
            r.f_hz,
            gx if np.isfinite(gx) else "",
            eff if np.isfinite(eff) else "",
            bx, by, bz,
            r.size_violation
        ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nGain summary CSV saved -> {path}")

# ===================== NEC EXPORT (4nec2-friendly) =====================
def nec_freq_block(f0_hz: float) -> str:
    f0_mhz = f0_hz / 1e6
    df_mhz = f0_mhz * NEC_FREQ_SPAN_FRAC * 2 / max(NEC_FREQ_POINTS-1, 1)
    f_start = f0_mhz * (1 - NEC_FREQ_SPAN_FRAC)
    # FR I1 Nsteps I3 I4 Fstart Df
    return f"FR 0 {NEC_FREQ_POINTS} 0 0 {f_start:.6f} {df_mhz:.6f}"

def nec_header(kind: str) -> List[str]:
    return [
        f"CM MAGPIE auto-generated {kind}",
        "CM Generated by MAGPIE Antenna Designer (Python)",
        "CM Load this file in 4nec2 / NEC2 engine.",
    ]

def nec_footer() -> List[str]:
    return ["EN"]

def write_nec_file(result: Result, lines: List[str], basename: str):
    if not EXPORT_NEC:
        return
    ensure_dir(NEC_DIR)
    path = os.path.join(NEC_DIR, f"{basename}.nec")
    with open(path, "w", encoding="ascii", errors="ignore") as f:
        for line in lines:
            f.write(line.rstrip() + "\r\n")
    print(f"NEC deck saved -> {path}  (lines: {len(lines)})")

def export_nec_for_result(result: Result):
    f0 = result.f_hz
    kind = result.kind
    lam = wavelength(f0)
    wire_radius = (result.build.get("wire_d_mm", WIRE_DIAM_MM_DEFAULT) / 2000.0)

    lines: List[str] = []
    lines += nec_header(kind)
    lines.append("CE")  # end of comments

    if kind == "dipole":
        L = result.build["total_length_m"]
        segs = 101
        lines.append(f"GW 1 {segs} 0 0 {-L/2:.6f} 0 0 {L/2:.6f} {wire_radius:.6f}")
        lines.append("GE 0")
        lines.append(nec_freq_block(f0))
        feed_seg = (segs+1)//2
        lines.append(f"EX 0 1 {feed_seg} 0 1.0 0.0")
        lines.append("RP 0 181 1 1000 0 180 1 0 0 1")
        lines += nec_footer()
        write_nec_file(result, lines, f"{kind}_build")

    elif kind == "monopole":
        h = result.build["height_m"]
        gp_r = result.build["gp_radius_m"]
        segs_vert = 101
        segs_rad = 11
        lines.append(f"GW 1 {segs_vert} 0 0 0 0 0 {h:.6f} {wire_radius:.6f}")
        for i, (x,y) in enumerate([(gp_r,0),(-gp_r,0),(0,gp_r),(0,-gp_r)], start=2):
            lines.append(f"GW {i} {segs_rad} 0 0 0 {x:.6f} {y:.6f} 0 {wire_radius:.6f}")
        lines.append("GN 1")
        lines.append("GE 0")
        lines.append(nec_freq_block(f0))
        feed_seg = 2
        lines.append(f"EX 0 1 {feed_seg} 0 1.0 0.0")
        lines.append("RP 0 91 1 1000 0 90 1 0 0 1")
        lines += nec_footer()
        write_nec_file(result, lines, f"{kind}_build")

    elif kind == "loop":
        diam = result.build["diam_m"]
        r = diam/2
        N_wires = 16
        segs_per_wire = 3
        tag = 1
        for i in range(N_wires):
            phi1 = 2*math.pi*i/N_wires
            phi2 = 2*math.pi*(i+1)/N_wires
            x1, y1 = r*math.cos(phi1), r*math.sin(phi1)
            x2, y2 = r*math.cos(phi2), r*math.sin(phi2)
            lines.append(
                f"GW {tag} {segs_per_wire} {x1:.6f} {y1:.6f} 0 {x2:.6f} {y2:.6f} 0 {wire_radius:.6f}"
            )
            tag += 1
        lines.append("GE 0")
        lines.append(nec_freq_block(f0))
        feed_seg = (segs_per_wire+1)//2
        lines.append(f"EX 0 1 {feed_seg} 0 1.0 0.0")
        lines.append("RP 0 181 1 1000 0 180 1 0 0 1")
        lines += nec_footer()
        write_nec_file(result, lines, f"{kind}_build")

    elif kind == "nmode_helix":
        h = result.build["height_m"]
        segs = 101
        lines.append(f"GW 1 {segs} 0 0 0 0 0 {h:.6f} {wire_radius:.6f}")
        lines.append("GE 0")
        lines.append(nec_freq_block(f0))
        feed_seg = 2
        lines.append(f"EX 0 1 {feed_seg} 0 1.0 0.0")
        lines.append("RP 0 181 1 1000 0 180 1 0 0 1")
        lines += nec_footer()
        write_nec_file(result, lines, f"{kind}_build")

    elif kind == "axial_helix":
        turns = result.build["turns"]
        D = result.build["diam_m"]
        S = result.build["spacing_m"]
        r = D/2
        N_points = 40 * turns
        z_total = S*turns
        pts = []
        for i in range(N_points+1):
            t = i / N_points
            phi = 2*math.pi*turns*t
            z = z_total * t
            x = r*math.cos(phi)
            y = r*math.sin(phi)
            pts.append((x,y,z))
        segs_per_wire = 1
        tag = 1
        for i in range(N_points):
            x1,y1,z1 = pts[i]
            x2,y2,z2 = pts[i+1]
            lines.append(
                f"GW {tag} {segs_per_wire} {x1:.6f} {y1:.6f} {z1:.6f} "
                f"{x2:.6f} {y2:.6f} {z2:.6f} {wire_radius:.6f}"
            )
            tag += 1
        lines.append("GE 0")
        lines.append(nec_freq_block(f0))
        lines.append("EX 0 1 1 0 1.0 0.0")
        lines.append("RP 0 181 1 1000 0 180 1 0 0 1")
        lines += nec_footer()
        write_nec_file(result, lines, f"{kind}_build")

    elif kind == "yagi":
        lengths = result.build["element_lengths_m"]
        pos = result.build["element_pos_along_boom_m"]
        segs = 21
        for i, (L, zpos) in enumerate(zip(lengths, pos), start=1):
            x1, x2 = -L/2, L/2
            lines.append(
                f"GW {i} {segs} {x1:.6f} 0 {zpos:.6f} {x2:.6f} 0 {zpos:.6f} {wire_radius:.6f}"
            )
        lines.append("GE 0")
        lines.append(nec_freq_block(f0))
        feed_seg = (segs+1)//2
        # element #2 is driven
        lines.append(f"EX 0 2 {feed_seg} 0 1.0 0.0")
        lines.append("RP 0 181 1 1000 0 180 1 0 0 1")
        lines += nec_footer()
        write_nec_file(result, lines, f"{kind}_build")

    elif kind == "dish":
        D = result.build["dish_diam_m"]
        f_focus = D/4
        L = 0.1 * lam
        segs = 31
        z0 = f_focus
        lines.append(
            f"GW 1 {segs} 0 0 {z0 - L/2:.6f} 0 0 {z0 + L/2:.6f} {wire_radius:.6f}"
        )
        lines.append("GE 0")
        lines.append(nec_freq_block(f0))
        feed_seg = (segs+1)//2
        lines.append(f"EX 0 1 {feed_seg} 0 1.0 0.0")
        lines.append("RP 0 181 1 1000 0 180 1 0 0 1")
        lines += nec_footer()
        write_nec_file(result, lines, f"{kind}_build")

# ===================== PIPELINE =====================
def compute_all():
    f0 = F_HZ_DEFAULT
    wire_mm = WIRE_DIAM_MM_DEFAULT

    antennas: List[Tuple[Result, Callable[[np.ndarray], np.ndarray], str]] = []

    dip   = design_dipole(f0, wire_mm)
    mono  = design_monopole(f0, wire_mm, GROUND_PLANE_RADIUS_M, RADIALS_COUNT)
    loop  = design_loop(f0, wire_mm, LOOP_DIAM_M_DEFAULT, LOOP_TURNS_DEFAULT)
    nmod  = design_nmode_helix(f0, wire_mm, HELIX_HEIGHT_M, HELIX_DIAM_M, HELIX_TURNS)
    axial = design_axial_helix(f0, AX_TURNS_DEFAULT, AX_SPACING_LAMBDA, AX_DIAM_LAMBDA)
    yagi  = design_yagi(f0, YAGI_DIR_COUNT, YAGI_BOOM_SPACING_L, YAGI_WIRE_DIAM_MM)
    dish  = design_dish(f0, DISH_DIAM_M, DISH_EFFICIENCY)

    antennas.extend([
        (dip,   pat_dipole,   "Dipole"),
        (mono,  pat_monopole, "Monopole"),
        (loop,  pat_loop,     "Small Loop"),
        (nmod,  pat_nmode,    "Normal-Mode Helix"),
        (axial, pat_axial,    "Axial Helix"),
        (yagi,  pat_yagi,     "Yagi"),
        (dish,  pat_dish,     "Dish"),
    ])

    results_only: List[Result] = []

    for res, _, _ in antennas:
        summarize(res)
        csvpath = write_csv_single(res, SAVE_DIR, f"{res.kind}_build")
        print(f"CSV saved -> {csvpath}")
        if EXPORT_NEC:
            export_nec_for_result(res)
        results_only.append(res)

    # After all antennas processed, write gain summary
    write_gain_summary_table(results_only)

    return antennas

def plot_all(antennas):
    ensure_dir(SAVE_DIR)
    for res, pattern_fn, label in antennas:
        plot_polar_theta(
            pattern_fn,
            f"{label} polar",
            os.path.join(SAVE_DIR, f"{res.kind}_polar.png")
        )
        plot_3d_pattern(
            pattern_fn,
            f"{label} 3D pattern",
            os.path.join(SAVE_DIR, f"{res.kind}_3d.png")
        )
        plot_smith(
            res.sweep_Zin,
            f"{label} Smith Chart (analytic)",
            os.path.join(SAVE_DIR, f"{res.kind}_smith.png")
        )
        plot_s11(
            res.sweep_freqs,
            res.sweep_Zin,
            f"{label} |S11| est (analytic)",
            os.path.join(SAVE_DIR, f"{res.kind}_s11.png")
        )

def main():
    ensure_dir(SAVE_DIR)
    if EXPORT_NEC:
        print(f"NEC export ENABLED. NEC files will be written under: {NEC_DIR}")
    else:
        print("NEC export DISABLED (EXPORT_NEC = False).")

    antennas = compute_all()
    plot_all(antennas)
    print("\nAll antennas processed. Plots generated and NEC decks exported for 4nec2 (if enabled).")

if __name__ == "__main__":
    main()
