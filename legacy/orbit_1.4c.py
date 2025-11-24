# ============================================================
# MAGPIE Mars Transfer "&" MOI Simulator
# 
# Liam Piper
# 2025

# ============================================================

import warnings
warnings.filterwarnings('ignore', message=r'ERFA function')

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from poliastro.bodies import Sun, Earth
from poliastro.iod.izzo import lambert
from poliastro.twobody import Orbit
from skyfield.api import load
import datetime
import os

# ===================== USER TOGGLES =====================
# Launch Vehicle selection:
USE_FALCON_HEAVY = False      # False = Falcon 9, True = Falcon Heavy (expendable)
LV_RESERVE_FRAC  = 0.10       # Fraction of LV Stage-2 propellant held as reserve (never touched)

# ---- LV2 underload model ----
LV2_ASCENT_SCALING   = "linear"   # "linear" or "sqrt"
LV2_PROP_TO_ORBIT_AT_CAP_FRAC = 0.78
ALLOW_LV2_POST_INSERTION_BURN = True   # let S2 help with LEO→C3 if it still has gas

# STAR kickstage:
USE_STAR           = False
STAR_MODE          = "fixed"   # "fixed" | "autosize" | "off"
AUTO_DISABLE_STAR  = True
FORCE_KEEP_STAR    = False

# Aerocapture + MOI shaping:
USE_AEROCAPTURE     = False    # Enable aerocapture option for MOI
MINIMUM_ORBIT_MODE  = False    # If True, skip dv3 circularize (we keep the big ellipse)

# Porkchop search (days from seed, and number of candidates to report):
PORKCHOP_SEARCH_DAYS = 600
PORKCHOP_LIST_TOP    = 12

# GO-ANYWAY behavior
GO_ANYWAY_IF_NO_C3   = True

# Plot control
SHOW_PLOTS = True

# --- OUTPUTS / LOGGING (new) ---
SAVE_CSV_ORIENTATION   = True    # write timeline_orient.csv with RTN + orientation labels
SAVE_JSON_ORIENTATION  = False   # optional small JSON summary

# --- PLOTS (new) ---
PLOT_ORIENTATION_TIMELINE = True # categorical “status vs time” plot (coast/pro/retro/normal/radial/EDL)
PLOT_CAPTURE_3D_ORIENTED  = True # new 3D, v∞-oriented initial-capture orbit plot (wicked cool)

# ===================== CONSTANTS & VEHICLE DATA =====================
g0      = 9.80665
mu_e    = Earth.k.to(u.m**3/u.s**2).value
mu_m    = 4.282837e13
R_e     = 6371e3
R_m     = 3389.5e3
h_LEO   = 350e3
h_MOI   = 250e3
GM_sun  = 1.32712440018e20

# Spacecraft (MONARC bus)
m_sc_wet = 7485.0    # kg   (includes MONARC prop when "full")
m_sc_dry = 3519.0    # kg
isp_monarc     = 235.0  # s
thrust_monarc  = 445.0  # N

# STAR kickstage (I think a star 26)
star_isp      = 292.0   # s
star_prop_max = 3596.0  # kg
star_dry      = 230.0   # kg

# Launch vehicle Stage-2 (approximate)
LV2_adaptor = 1750.0                # Extra Gubins (interstage, adapters, lines, etc.)
LV2_ISP     = 348.0                 # Merlin Vac ~348 s
LV2_PROP    = 110000.00              # kg prop in stage-2 (generic) [LOX+RP-1 total] 110000.00
LV2_DRY     = 4500.0 + LV2_adaptor  # kg dry stage-2
LEO_CAP_F9  = 22800.0               # kg
LEO_CAP_FH  = 63800.0               # kg

# --- LV2 prop split (LOX / RP-1) ---
LV2_OX_FUEL_MR = 2.56  # O/F mass ratio for Merlin Vac (ballpark)
LV2_FRAC_OX   = LV2_OX_FUEL_MR / (LV2_OX_FUEL_MR + 1.0)
LV2_FRAC_FUEL = 1.0 / (LV2_OX_FUEL_MR + 1.0)

# Mission geometry
EI_ALT_M            = 250e3
PERIAPSIS_ALT_M     = 40e3
PLANE_CHANGE_DEG    = 120.0
APOAPSIS_RATIO      = 150.0         # initial big ellipse ratio (apo/peri)

# Burn discretization (perigee-pass “chunks”)
PER_PASS_TIME_MIN_DEP = 10.0
PER_PASS_TIME_MIN_MOI = 6.0
MAX_PASSES_PER_PHASE  = 200

# Porkchop seed date
depart_dt_seed = datetime.datetime(2028, 11, 10, tzinfo=datetime.timezone.utc)

# ===================== AEROCAPTURE PHYSICS TOGGLES =====================
AEROCAPTURE_PHYSICS  = True     # if True, replace energy bookkeeping with atmosphere simulation
ENTRY_USE_LIFT       = False    # False=ballistic; True=lifting entry (basic lift only)
ENTRY_BANK_DEG       = 0.0      # constant bank (deg)
CL                   = 0.3      # set ~0.2–0.4 if ENTRY_USE_LIFT=True
CD                   = 1.5
REF_AREA_M2          = 12.0
NOSE_RADIUS_M        = 0.5      # m (Sutton–Graves radius)
SG_K                 = 1.2e-4   # Sutton–Graves K: q̇ = K √(ρ/Rn) V^3   (SI)

# Atmosphere model selection: "exponential" | "layered_exp" | "tabulated"
ATMOSPHERE_MODEL     = "layered_exp"
# density scaling factor to emulate season/lat (applies to all models)
ATM_DENSITY_SCALE    = 1.0

# Exponential model params
MARS_RHO0            = 0.020   # kg/m^3 near surface
MARS_SCALE_HEIGHT    = 11100.0 # m

# Layered exponential (alt_km upper bounds, and scale heights per layer in m)
_LAY_ALT_KM   = np.array([7, 20, 40, 60, 80, 120, 160, 200, 300], dtype=float)
_LAY_H_M      = np.array([10e3, 11e3, 12e3, 13e3, 15e3, 18e3, 22e3, 30e3, 40e3], dtype=float)
_LAY_RHO0     = 0.020  # surface reference; layers are continuous via cumulative construction below

# Tabulated profile (coarse, mean atmosphere; alt[km], rho[kg/m^3], T[K])
_TAB_ALT_KM = np.array(
    [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300],
    dtype=float
)
_TAB_RHO = np.array(
    [0.0200, 0.0155, 0.0120, 0.0092, 0.0070, 0.0053, 0.0040, 0.0023, 0.0013, 7.5e-4,
     4.4e-4, 2.5e-4, 1.4e-4, 8.0e-5, 3.0e-5, 1.2e-5, 5.5e-6, 2.6e-6, 1.3e-6, 4.0e-7, 1.5e-7],
    dtype=float
)
_TAB_TEMP = np.array(
    [210, 205, 200, 195, 190, 185, 180, 175, 170, 165,
     160, 155, 150, 150, 150, 150, 150, 150, 150, 150, 150],
    dtype=float
)

# Entry integration
ENTRY_DT             = 0.25    # s
ENTRY_MAX_DURATION   = 3*3600  # s
ENTRY_FPA_SOLVE      = True    # shoot for target periapsis altitude
FPA_BOUNDS_DEG       = (-15.0, -1.0)
FPA_SOLVE_TOL_M      = 1000.0  # 1 km

# Safety / design limits (quick sanity bounders)
Q_LIMIT_PA           = 30e3     # 30 kPa
HEAT_LIMIT_WM2       = 1.0e6    # 1 MW/m^2
G_LIMIT              = 8.0      # 8 g

# ===================== HELPERS =====================
def _as_quantity_kms(x):
    if hasattr(x, "unit"):
        return x.to(u.km/u.s)
    return np.asarray(x) * (u.km/u.s)

def vec_norm(q):
    if hasattr(q, "unit"):
        arr = q.to(u.km/u.s).value
        return np.sqrt(np.sum(arr*arr)) * (u.km/u.s)
    arr = np.asarray(q)
    return np.sqrt(np.sum(arr*arr))

def solve_lambert(mu, r0, r1, tof):
    res = lambert(mu, r0, r1, tof)
    import types
    try:
        if isinstance(res, types.GeneratorType):
            first = next(res)
            if isinstance(first, tuple) and len(first) == 2:
                return _as_quantity_kms(first[0]), _as_quantity_kms(first[1])
            if hasattr(first, "v0") and hasattr(first, "v1"):
                return _as_quantity_kms(first.v0), _as_quantity_kms(first.v1)
            arr = np.array(first)
            if arr.shape == (2, 3):
                return _as_quantity_kms(arr[0]), _as_quantity_kms(arr[1])
            raise RuntimeError("Unsupported lambert() generator element type")
    except StopIteration:
        raise RuntimeError("lambert() generator yielded no solutions")
    if isinstance(res, tuple) and len(res) == 2:
        return _as_quantity_kms(res[0]), _as_quantity_kms(res[1])
    if hasattr(res, "v0") and hasattr(res, "v1"):
        return _as_quantity_kms(res.v0), _as_quantity_kms(res.v1)
    if isinstance(res, (list, tuple)) and len(res) > 0:
        first = res[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            return _as_quantity_kms(first[0]), _as_quantity_kms(first[1])
    arr = np.array(res)
    if arr.shape == (2, 3):
        return _as_quantity_kms(arr[0]), _as_quantity_kms(arr[1])
    raise RuntimeError(f"Unsupported lambert() return type: {type(res)}")

def rocket_eq_delta_v(isp_s, m0, mf):
    if mf <= 0 or mf >= m0: return 0.0
    return isp_s * g0 * np.log(m0 / mf)

def rocket_eq_prop_for_dv(isp_s, m_dry, dv_mps, m0_guess=None):
    mass_ratio = np.exp(dv_mps / (isp_s * g0))
    m0_needed  = mass_ratio * m_dry
    return max(0.0, m0_needed - m_dry)

def perigee_pass_burns_required(dv_target_kms, m0, isp_s, thrust_n, per_pass_min, max_passes, phase_name=""):
    dv_target_mps = dv_target_kms * 1000.0
    if dv_target_mps <= 1e-9:
        return [], 0.0, m0, 0.0
    passes = []
    delivered = 0.0
    t_total_min = 0.0
    m = m0
    dt = per_pass_min * 60.0
    for i in range(1, max_passes+1):
        a = thrust_n / m
        dv_mps = a * dt
        dm = m * (1.0 - np.exp(-dv_mps / (isp_s * g0)))
        m_new = m - dm
        delivered += dv_mps
        t_total_min += per_pass_min
        passes.append({
            "pass": i,
            "dv_kms": dv_mps/1000.0,
            "time_min": per_pass_min,
            "m_start": m,
            "m_end": m_new,
            "m_used": dm
        })
        m = m_new
        if delivered >= dv_target_mps - 1e-9:
            break
    return passes, delivered/1000.0, m, t_total_min

def _savefig(name):
    try:
        plt.savefig(f"{name}.png", dpi=130)
    except Exception as e:
        print("[plot save warning]", name, e)

# --- New helpers for RTN and labeling ---
def _rtn_basis(r_vec_km, v_vec_kmps):
    """Return RTN (Radial, Transverse, Normal) unit vectors from state (heliocentric)."""
    r = np.asarray(r_vec_km); v = np.asarray(v_vec_kmps)
    Rhat = r / (np.linalg.norm(r) + 1e-15)
    h    = np.cross(r, v)
    Nhat = h / (np.linalg.norm(h) + 1e-15)
    That = np.cross(Nhat, Rhat)     # completes right-handed RTN
    return Rhat, That, Nhat

# ===================== EPHEMERIDES =====================
planets = load('de421.bsp')
sun   = planets['sun']
earth = planets['earth']
mars  = planets['mars']
ts    = load.timescale()

# ================== PORKCHOP (yum) ===================
def porkchop_candidates(seed_dt, search_days, top_n):
    results = []
    for dd in range(0, search_days, 2):
        dep_dt = seed_dt + datetime.timedelta(days=dd)
        t0 = ts.utc(dep_dt.year, dep_dt.month, dep_dt.day, 0, 0, 0)

        rE_helio = (earth - sun).at(t0).position.km * 1e3
        rM_helio_guess = (mars - sun).at(t0).position.km * 1e3
        rE_norm  = np.linalg.norm(rE_helio)
        rM_norm  = np.linalg.norm(rM_helio_guess)
        a_trans  = 0.5 * (rE_norm + rM_norm)
        tof_s    = np.pi * np.sqrt(a_trans**3 / GM_sun)
        arr_dt   = dep_dt + datetime.timedelta(seconds=float(tof_s))
        t1 = ts.utc(arr_dt.year, arr_dt.month, arr_dt.day, 0, 0, 0)

        r0_km = (earth - sun).at(t0).position.km * u.km
        r1_km = (mars  - sun).at(t1).position.km * u.km
        try:
            v_dep, v_arr = solve_lambert(Sun.k, r0_km, r1_km, tof_s * u.s)
        except Exception:
            continue
        vE_kms = (earth - sun).at(t0).velocity.km_per_s * (u.km/u.s)
        v_inf_dep = vec_norm(v_dep - vE_kms).value

        r_LEO = R_e + h_LEO
        v_circ_e = np.sqrt(mu_e / r_LEO)
        v_esc_e  = np.sqrt(2.0 * mu_e / r_LEO)
        v_inf_m  = v_inf_dep * 1000.0
        dv_dep_mps = np.sqrt(v_inf_m**2 + v_esc_e**2) - v_circ_e
        results.append({
            "dep": dep_dt,
            "arr": arr_dt,
            "tof_days": int(tof_s//86400),
            "v_inf_dep": v_inf_dep,
            "dv_dep_kms": dv_dep_mps/1000.0
        })
    results.sort(key=lambda x: x["dv_dep_kms"])
    return results[:top_n]

# ------------- globals to feed plots -------------
_g_dv_lv2_used_kms = 0.0
_g_dv_star_used_kms = 0.0
_g_dv_monarc_dep_kms = 0.0

# ===================== ATMOSPHERE & ENTRY (unchanged) =====================
def atm_density_temp(h_m):
    h_km = max(0.0, h_m) / 1000.0
    if ATMOSPHERE_MODEL == "exponential":
        rho = MARS_RHO0 * np.exp(-h_m / MARS_SCALE_HEIGHT)
        T   = 200.0
        return ATM_DENSITY_SCALE * rho, T
    if ATMOSPHERE_MODEL == "layered_exp":
        alt_edges = np.concatenate(([0.0], _LAY_ALT_KM))
        Hs = _LAY_H_M
        rho = _LAY_RHO0
        base_rho = [_LAY_RHO0]
        for i in range(len(Hs)):
            dh = (alt_edges[i+1] - alt_edges[i]) * 1000.0
            rho = rho * np.exp(-dh / Hs[i])
            base_rho.append(rho)
        idx = np.searchsorted(_LAY_ALT_KM, h_km)
        idx = min(idx, len(Hs)-1)
        h0_km = alt_edges[idx]
        rho0_layer = base_rho[idx]
        H = Hs[idx]
        dh = (h_km - h0_km) * 1000.0
        rho = rho0_layer * np.exp(-dh / H)
        T = 210.0 - 0.5*h_km
        T = max(145.0, min(230.0, T))
        return ATM_DENSITY_SCALE * rho, T
    rho = np.interp(h_km, _TAB_ALT_KM, _TAB_RHO, left=_TAB_RHO[0], right=_TAB_RHO[-1])
    T   = np.interp(h_km, _TAB_ALT_KM, _TAB_TEMP, left=_TAB_TEMP[0], right=_TAB_TEMP[-1])
    return ATM_DENSITY_SCALE * rho, T

def sutton_graves_heat_flux_Wm2(rho, V, Rn=NOSE_RADIUS_M, K=SG_K):
    Rn_eff = max(Rn, 1e-4)
    return K * np.sqrt(max(rho, 0.0) / Rn_eff) * (V**3)

def entry_dynamics(t, state, planet_mu=mu_m, R_body=R_m, CD=CD, CL=CL, A=REF_AREA_M2,
                   m=m_sc_wet, bank_deg=ENTRY_BANK_DEG, use_lift=ENTRY_USE_LIFT):
    r, v, gamma = state
    h = r - R_body
    rho, _T = atm_density_temp(h)
    q   = 0.5 * rho * v*v
    D   = q * CD * A
    L   = (q * CL * A) if use_lift else 0.0
    bank = np.deg2rad(bank_deg)
    L_n  = L * np.cos(bank)
    g    = planet_mu / (r*r)

    r_dot     = v * np.sin(gamma)
    v_dot     = -(D/m) - g * np.sin(gamma)
    gamma_dot = 0.0
    if v > 1e-3:
        gamma_dot = (L_n/(m*v)) + (v/r - g/v) * np.cos(gamma)

    heat_Wm2 = sutton_graves_heat_flux_Wm2(rho, v)
    a_mag    = np.sqrt((D/m + g*np.sin(gamma))**2 + (L_n/(m))**2)

    return np.array([r_dot, v_dot, gamma_dot]), {
        "rho": rho, "q": q, "D": D, "L": L_n, "heat": heat_Wm2, "a": a_mag
    }

def rk4_step(fun, t, y, dt):
    k1, d1 = fun(t, y)
    k2, d2 = fun(t+0.5*dt, y + 0.5*dt*k1)
    k3, d3 = fun(t+0.5*dt, y + 0.5*dt*k2)
    k4, d4 = fun(t+dt,     y + dt*k3)
    y_next = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y_next, d4

def simulate_entry_pass(v_inf_arr_mps, gamma0_deg, rp_target_alt_m,
                        EI_alt_m=EI_ALT_M, dt=ENTRY_DT, max_dur=ENTRY_MAX_DURATION,
                        R_body=R_m, mu=mu_m, m=m_sc_wet):
    r_EI = R_body + EI_alt_m
    v_EI = np.sqrt(v_inf_arr_mps**2 + 2*mu / r_EI)
    gamma0 = np.deg2rad(gamma0_deg)

    t = 0.0
    y = np.array([r_EI, v_EI, gamma0], dtype=float)

    max_q = 0.0; max_g = 0.0; max_heat = 0.0; heat_int = 0.0
    min_alt = EI_alt_m; atmo_dv = 0.0

    hist_t=[]; hist_alt=[]; hist_v=[]; hist_q=[]; hist_g=[]; hist_heat=[]

    last_v = v_EI
    exited = False
    while t < max_dur:
        y, diag = rk4_step(lambda tt, yy: entry_dynamics(tt, yy, planet_mu=mu, R_body=R_body,
                                                         CD=CD, CL=CL, A=REF_AREA_M2, m=m,
                                                         bank_deg=ENTRY_BANK_DEG, use_lift=ENTRY_USE_LIFT),
                           t, y, dt)
        r, v, gamma = y
        alt = r - R_body
        if alt < 0: break

        q = diag["q"]; gload = diag["a"] / g0; heat = diag["heat"]
        max_q   = max(max_q, q)
        max_g   = max(max_g, gload)
        max_heat= max(max_heat, heat)
        heat_int += heat * dt
        min_alt = min(min_alt, alt)
        atmo_dv += max(0.0, last_v - v)
        last_v = v

        hist_t.append(t); hist_alt.append(alt); hist_v.append(v)
        hist_q.append(q); hist_g.append(gload); hist_heat.append(heat)

        if alt > EI_alt_m and t > 10.0:
            exited = True
            break
        t += dt

    results = {
        "time_s": np.array(hist_t),
        "alt_m": np.array(hist_alt),
        "v_mps": np.array(hist_v),
        "q_Pa": np.array(hist_q),
        "g_load": np.array(hist_g),
        "heat_Wm2": np.array(hist_heat),
        "max_q_Pa": max_q,
        "max_g": max_g,
        "max_heat_Wm2": max_heat,
        "heat_load_Jm2": heat_int,
        "min_alt_m": min_alt,
        "atmo_dv_mps": atmo_dv,
        "exited": exited,
        "gamma0_deg": gamma0_deg
    }
    return results

def solve_entry_fpa_for_target(v_inf_arr_mps, rp_target_alt_m, fpa_bounds_deg=FPA_BOUNDS_DEG,
                               tol_m=FPA_SOLVE_TOL_M, max_iter=18):
    lo, hi = fpa_bounds_deg
    r_lo = simulate_entry_pass(v_inf_arr_mps, lo, rp_target_alt_m)
    r_hi = simulate_entry_pass(v_inf_arr_mps, hi, rp_target_alt_m)
    def alt_err(res): return (res["min_alt_m"] - rp_target_alt_m)

    e_lo = alt_err(r_lo); e_hi = alt_err(r_hi)
    if e_lo*e_hi > 0:
        return r_lo if abs(e_lo) < abs(e_hi) else r_hi

    best = r_lo if abs(e_lo) < abs(e_hi) else r_hi
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        r_mid = simulate_entry_pass(v_inf_arr_mps, mid, rp_target_alt_m)
        e_mid = alt_err(r_mid)
        if abs(e_mid) < tol_m:
            return r_mid
        if e_lo*e_mid <= 0:
            hi = mid; r_hi = r_mid; e_hi = e_mid
        else:
            lo = mid; r_lo = r_mid; e_lo = e_mid
        best = r_mid if abs(e_mid) < abs(alt_err(best)) else best
    return best

# ===================== MAIN RUN =====================
def main():
    global _g_dv_lv2_used_kms, _g_dv_star_used_kms, _g_dv_monarc_dep_kms

    # --- Porkchop search ---
    cands = porkchop_candidates(depart_dt_seed, PORKCHOP_SEARCH_DAYS, PORKCHOP_LIST_TOP)
    print("\n=== Porkchop Top (by departure Δv only) ===")
    for i, c in enumerate(cands, 1):
        print(f"{i:2d}. dep {c['dep'].date()} | TOF {c['tof_days']:3d} d | v∞dep {c['v_inf_dep']:.3f} | "
              f"dep Δv {c['dv_dep_kms']:.3f} km/s")
    best = cands[0]
    depart_dt = best["dep"]
    arrival_dt = best["arr"]
    tof_s   = (arrival_dt - depart_dt).total_seconds()

    print(f"\n[Auto-apply] Using dep {depart_dt.date()} → arr {arrival_dt.date()} from porkchop best.\n")

    t0 = ts.utc(depart_dt.year, depart_dt.month, depart_dt.day, 0, 0, 0)
    t1 = ts.utc(arrival_dt.year, arrival_dt.month, arrival_dt.day, 0, 0, 0)
    r0_km   = (earth - sun).at(t0).position.km * u.km
    r1_km   = (mars  - sun).at(t1).position.km * u.km
    v_dep, v_arr = solve_lambert(Sun.k, r0_km, r1_km, tof_s * u.s)
    vE_kms = (earth - sun).at(t0).velocity.km_per_s * (u.km/u.s)
    vM_kms = (mars  - sun).at(t1).velocity.km_per_s * (u.km/u.s)
    v_inf_dep = vec_norm(v_dep - vE_kms)   # km/s
    v_inf_arr = vec_norm(v_arr - vM_kms)   # km/s

    # Required LEO→C3 (from earth)
    r_LEO    = R_e + h_LEO
    v_circ_e = np.sqrt(mu_e / r_LEO)
    v_esc_e  = np.sqrt(2.0 * mu_e / r_LEO)
    v_inf_m  = v_inf_dep.to(u.m/u.s).value
    dv_dep_mps = np.sqrt(v_inf_m**2 + v_esc_e**2) - v_circ_e
    c3 = (v_inf_dep.value)**2  # km^2/s^2

    # ===== Print high-level transfer numbers =====
    days  = int(tof_s // 86400)
    hours = int((tof_s % 86400) // 3600)
    mins  = int((tof_s % 3600) // 60)
    print(f"TOF: {days}d {hours}h {mins}m")
    print(f"C3 (km^2/s^2):                {c3:,.3f}")
    print(f"Departure v_inf:              {v_inf_dep.value:.3f} km/s")
    print(f"Departure burn Δv (LEO→C3):   {dv_dep_mps/1000.0:.3f} km/s")

    # ===== Assemble stack at LEO =====
    stack_mass = m_sc_wet
    star_prop = 0.0
    star_used = False
    star_auto_dropped = False

    if USE_STAR and STAR_MODE != "off":
        if STAR_MODE == "fixed":
            star_prop = min(star_prop_max, star_prop_max)
        elif STAR_MODE == "autosize":
            star_prop = star_prop_max
        stack_mass += (star_prop + star_dry)

    leo_cap = LEO_CAP_FH if USE_FALCON_HEAVY else LEO_CAP_F9
    lv_name = "FH" if USE_FALCON_HEAVY else "F9"
    if stack_mass > leo_cap:
        print(f"[ABORT] Stack @LEO {stack_mass:.1f} kg exceeds {lv_name} LEO capacity {leo_cap:.1f} kg.")
        return

    # ==== LV2 Underload Model (mixture-aware) ====
    payload_frac = max(0.0, min(1.0, stack_mass / leo_cap))
    exp = 1.0 if LV2_ASCENT_SCALING.lower().startswith("lin") else 0.5

    prop_to_orbit_at_cap = LV2_PROP_TO_ORBIT_AT_CAP_FRAC * LV2_PROP
    prop_to_orbit_actual = prop_to_orbit_at_cap * (payload_frac ** exp)
    prop_to_orbit_actual = max(0.0, min(LV2_PROP, prop_to_orbit_actual))

    prop_reserve_total = max(0.0, min(LV2_PROP, LV_RESERVE_FRAC * LV2_PROP))
    prop_leftover_total = max(0.0, LV2_PROP - prop_to_orbit_actual - prop_reserve_total)

    lox_total   = LV2_PROP * LV2_FRAC_OX
    fuel_total  = LV2_PROP * LV2_FRAC_FUEL
    lox_to_orbit   = prop_to_orbit_actual * LV2_FRAC_OX
    fuel_to_orbit  = prop_to_orbit_actual * LV2_FRAC_FUEL
    lox_reserve    = prop_reserve_total * LV2_FRAC_OX
    fuel_reserve   = prop_reserve_total * LV2_FRAC_FUEL
    lox_leftover   = prop_leftover_total * LV2_FRAC_OX
    fuel_leftover  = prop_leftover_total * LV2_FRAC_FUEL

    print("\n[LV Stage-2 Ascent Ledger]")
    print(f"  Vehicle: {lv_name}  | LEO cap: {leo_cap:.0f} kg")
    print(f"  Payload(stack) mass @LEO: {stack_mass:.1f} kg  ({payload_frac*100:.1f}% of cap)")
    print(f"  LV2 tank total: {LV2_PROP:.0f} kg  | dry: {LV2_DRY:.0f} kg  | ISP: {LV2_ISP:.1f} s")
    print(f"  Mix (O/F): {LV2_OX_FUEL_MR:.2f}  → LOX {LV2_FRAC_OX*100:.1f}% / RP-1 {LV2_FRAC_FUEL*100:.1f}%")
    print(f"  To reach LEO (model):  {prop_to_orbit_actual:.0f} kg  [at-cap {prop_to_orbit_at_cap:.0f} kg; scaling={LV2_ASCENT_SCALING}]")
    print(f"     • LOX:  {lox_to_orbit:.0f} kg   • RP-1: {fuel_to_orbit:.0f} kg")
    print(f"  Held as reserve:       {prop_reserve_total:.0f} kg  (LOX {lox_reserve:.0f} / RP-1 {fuel_reserve:.0f})")
    print(f"  Post-insertion leftover: {prop_leftover_total:.0f} kg  (LOX {lox_leftover:.0f} / RP-1 {fuel_leftover:.0f})")

    # LV2 post-insertion burn toward C3 — mixture-aware
    lv2_dv_used_mps = 0.0
    lv2_prop_burned_now = 0.0
    stack_after_lv = stack_mass

    if ALLOW_LV2_POST_INSERTION_BURN and (lox_leftover > 1e-6 and fuel_leftover > 1e-6):
        m0_all  = LV2_DRY + stack_mass + (lox_leftover + fuel_leftover)
        mf_all  = LV2_DRY + stack_mass
        dv_full = rocket_eq_delta_v(LV2_ISP, m0_all, mf_all)

        dv_target = min(dv_full, dv_dep_mps)

        m0_need = LV2_DRY + stack_mass + (lox_leftover + fuel_leftover)
        mf_need = m0_need / np.exp(dv_target / (LV2_ISP * g0))
        prop_needed = max(0.0, m0_need - mf_need)

        lox_need  = prop_needed * LV2_FRAC_OX
        fuel_need = prop_needed * LV2_FRAC_FUEL

        scale = min(1.0,
                    lox_leftover / max(1e-9, lox_need),
                    fuel_leftover / max(1e-9, fuel_need))
        lox_burn  = lox_need  * scale
        fuel_burn = fuel_need * scale
        prop_burn = lox_burn + fuel_burn

        m0_actual = LV2_DRY + stack_mass + (lox_leftover + fuel_leftover)
        mf_actual = m0_actual - prop_burn
        dv_actual = rocket_eq_delta_v(LV2_ISP, m0_actual, mf_actual)

        lox_leftover  = max(0.0, lox_leftover  - lox_burn)
        fuel_leftover = max(0.0, fuel_leftover - fuel_burn)
        lv2_prop_burned_now = prop_burn
        lv2_dv_used_mps = min(dv_actual, dv_dep_mps)
        stack_after_lv = mf_actual - LV2_DRY

        print(f"\n[LV2 → C3] Using leftover toward LEO→C3 (mixture-aware):")
        print(f"  LV2 Δv possible (all leftover): {dv_full/1000.0:.3f} km/s")
        print(f"  LV2 Δv requested:               {dv_target/1000.0:.3f} km/s")
        if scale < 0.999:
            print("  [mixture-limited] One component hit zero before target Δv.")
        print(f"  LV2 Δv achieved:                {lv2_dv_used_mps/1000.0:.3f} km/s")
        print(f"  Prop burned now:                {lv2_prop_burned_now:.0f} kg  (LOX {lox_burn:.0f} / RP-1 {fuel_burn:.0f})")
        m0_chk = LV2_DRY + stack_mass + (lox_leftover + fuel_leftover + lv2_prop_burned_now)
        mf_chk = LV2_DRY + stack_mass + (lox_leftover + fuel_leftover)
        dv_chk = rocket_eq_delta_v(LV2_ISP, m0_chk, mf_chk)
        print(f"  [check] Δv from prop used: {dv_chk/1000.0:.3f} km/s (≈ LV2 Δv achieved)")
    else:
        stack_after_lv = stack_mass
        if not ALLOW_LV2_POST_INSERTION_BURN:
            print("\n[LV2 → C3] Post-insertion burn disabled by toggle.")
        else:
            print("\n[LV2 → C3] No usable leftover (one or both components empty).")

    lv2_prop_final_total = (lox_reserve + fuel_reserve) + (lox_leftover + fuel_leftover)
    lv2_prop_final_pct   = 100.0 * lv2_prop_final_total / LV2_PROP if LV2_PROP > 0 else 0.0
    lox_final   = lox_reserve + lox_leftover
    fuel_final  = fuel_reserve + fuel_leftover

    print("\n[LV2 Fuel Status]")
    print(f"  Final LV2 remaining (total): {lv2_prop_final_total:.0f} kg  ({lv2_prop_final_pct:.2f}% of tank)")
    print(f"     • LOX:  {lox_final:.0f} kg  ({100.0*lox_final/lox_total:.2f}% of LOX)")
    print(f"     • RP-1: {fuel_final:.0f} kg ({100.0*fuel_final/fuel_total:.2f}% of RP-1)")

    _g_dv_lv2_used_kms = lv2_dv_used_mps / 1000.0
    dv_dep_remain_mps = max(0.0, dv_dep_mps - lv2_dv_used_mps)

    # ===== STAR logic =====
    star_dv_used_mps = 0.0
    if dv_dep_remain_mps > 1e-6 and USE_STAR and STAR_MODE != "off":
        m_start = stack_after_lv
        dv_star_all = 0.0
        if star_prop > 0.0:
            m_end_if_all = m_start - star_prop
            dv_star_all = rocket_eq_delta_v(star_isp, m_start, m_end_if_all)
        dv_to_use = min(dv_dep_remain_mps, dv_star_all)
        if dv_to_use > 1e-6:
            mf_req = m_start / np.exp(dv_to_use / (star_isp * g0))
            star_prop_used = (m_start - mf_req)
            star_prop = max(0.0, star_prop - star_prop_used)
            star_used = True
            stack_after_lv = mf_req
            star_dv_used_mps = dv_to_use
            print(f"\n[STAR → C3] Δv used: {star_dv_used_mps/1000.0:.3f} km/s | prop burned: {star_prop_used:.0f} kg")
        if star_used or (STAR_MODE in ["fixed","autosize"] and not FORCE_KEEP_STAR):
            stack_after_lv -= star_dry
        else:
            if AUTO_DISABLE_STAR and dv_dep_remain_mps <= 1e-6 and not FORCE_KEEP_STAR:
                stack_after_lv -= (star_prop + star_dry)
                star_auto_dropped = True
    else:
        if USE_STAR and STAR_MODE != "off" and AUTO_DISABLE_STAR and not FORCE_KEEP_STAR and dv_dep_remain_mps <= 1e-6:
            stack_after_lv -= (star_prop + star_dry)
            star_auto_dropped = True

    _g_dv_star_used_kms = star_dv_used_mps / 1000.0
    dv_dep_remain_mps = max(0.0, dv_dep_remain_mps - star_dv_used_mps)

    # Departure ledger 
    print("\n[Departure Δv Ledger]")
    print(f"  Required LEO→C3 (raw):   {dv_dep_mps/1000.0:.3f} km/s")
    print(f"  LV2 leftover provided:    {_g_dv_lv2_used_kms:.3f} km/s")
    if USE_STAR and STAR_MODE != "off":
        if star_auto_dropped:
            print("  STAR: not required — auto-dropped from stack.")
        elif _g_dv_star_used_kms > 0:
            print(f"  STAR provided:            {_g_dv_star_used_kms:.3f} km/s")
        else:
            print("  STAR present, no Δv used.")
    else:
        print("  STAR: disabled.")
    print(f"  Remaining for MONARC:     {dv_dep_remain_mps/1000.0:.3f} km/s")

    monarc_prop_full = m_sc_wet - m_sc_dry

    # MONARC departure perigee-pass burns if needed
    dep_passes, dep_deliv_kms, m_after_dep, dep_t_min = [], 0.0, None, 0.0
    monarc_prop_used_dep = 0.0
    if dv_dep_remain_mps > 1e-6:
        dep_passes, dep_deliv_kms, m_after_dep, dep_t_min = perigee_pass_burns_required(
            dv_dep_remain_mps/1000.0, m_sc_wet, isp_monarc, thrust_monarc,
            PER_PASS_TIME_MIN_DEP, MAX_PASSES_PER_PHASE, "Departure MONARC"
        )
        monarc_prop_used_dep = m_sc_wet - m_after_dep
        print("\n[MONARC (LEO→C3)] Perigee-pass burns:")
        for p in dep_passes[:80]:
            print(f"  pass {p['pass']:02d}: Δv={p['dv_kms']:.4f} km/s, time={p['time_min']:.2f} min, "
                  f"mass {p['m_start']:.1f}→{p['m_end']:.1f} kg (used {p['m_used']:.1f} kg)")
        if len(dep_passes) > 80:
            print(f"  ... {len(dep_passes)-80} more passes not shown")
        if dep_deliv_kms >= dv_dep_remain_mps/1000.0 - 1e-6:
            print("\n--- MONARC Departure Summary ---")
            print(f"Delivered Δv:   {dep_deliv_kms:.3f} km/s in {len(dep_passes)} passes ({dep_t_min:.1f} min)")
            print(f"Prop used:      {monarc_prop_used_dep:.1f} kg ({100*monarc_prop_used_dep/monarc_prop_full:.1f}% of MONARC prop)")
    else:
        m_after_dep = m_sc_wet

    _g_dv_monarc_dep_kms = dep_deliv_kms

    achieved_escape = (dv_dep_remain_mps <= 1e-6) or (dep_deliv_kms >= dv_dep_remain_mps/1000.0 - 1e-6)
    if not achieved_escape and not GO_ANYWAY_IF_NO_C3:
        print("\n=== FINAL STATUS: Escape NOT achieved ===")
        print(f"Required dep Δv (net): {(dv_dep_remain_mps/1000.0):.3f} km/s")
        print(f"Delivered by MONARC:    {dep_deliv_kms:.3f} km/s")
        shortfall = max(0.0, dv_dep_remain_mps/1000.0 - dep_deliv_kms)
        print(f"Shortfall:              {shortfall:.3f} km/s")
        print("Consider enabling STAR or switching to FH, or widening porkchop search.")
        return

    if not achieved_escape and GO_ANYWAY_IF_NO_C3:
        r_p = R_e + h_LEO
        v_p = np.sqrt(mu_e / r_p)
        v_p_new = v_p + dep_deliv_kms*1000.0
        a_new = 1.0 / (2.0/r_p - (v_p_new**2)/mu_e)
        e_new = 1.0 - r_p / a_new
        r_a = a_new * (1+e_new)
        print("\n=== FINAL STATUS: Escape NOT achieved ===")
        print("Earth-bound orbit achieved:")
        print(f"  rp = {r_p/1000.0:,.1f} km (alt {h_LEO/1000.0:,.1f} km)")
        print(f"  ra = {r_a/1000.0:,.1f} km (alt {r_a/1000.0 - R_e/1000.0:,.1f} km)")
        print(f"  a  = {a_new/1000.0:,.1f} km, e = {e_new:.6f}")
        T = 2*np.pi*np.sqrt(a_new**3 / mu_e)
        print(f"  Period = {T/3600.0:.2f} h")
        make_all_plots(depart_dt, arrival_dt, t0, tof_s, r0_km, v_dep, vE_kms, False, False,
                       final_orbit=("Earth", r_p, r_a),
                       dv_breakdown=(_g_dv_lv2_used_kms, _g_dv_star_used_kms, _g_dv_monarc_dep_kms),
                       entry_results=None,
                       initial_capture_orbit=None)  # no Mars here
        return

    # =============== Proceed to MOI (we reached escape) ===============
    transfer0 = Orbit.from_vectors(Sun, r0_km, v_dep)
    rE0 = (earth - sun).at(t0).position.km * u.km
    vE0 = (earth - sun).at(t0).velocity.km_per_s * (u.km/u.s)
    rM0 = (mars  - sun).at(t0).position.km * u.km
    vM0 = (mars  - sun).at(t0).velocity.km_per_s * (u.km/u.s)
    earth_orb0 = Orbit.from_vectors(Sun, rE0, vE0)
    mars_orb0  = Orbit.from_vectors(Sun, rM0, vM0)

    # target initial capture ellipse
    r_p = R_m + h_MOI
    r_a = APOAPSIS_RATIO * r_p
    a   = 0.5 * (r_p + r_a)
    v_esc_m     = np.sqrt(2.0 * mu_m / r_p)
    v_hyp_p     = np.sqrt(v_esc_m**2 + (v_inf_arr.to(u.m/u.s).value)**2)
    v_peri_ellip= np.sqrt(mu_m * (2.0/r_p - 1.0/a))
    v_apo_ellip = np.sqrt(mu_m * (2.0/r_a - 1.0/a))
    v_circ_m    = np.sqrt(mu_m / r_p)

    theta   = np.deg2rad(PLANE_CHANGE_DEG)

    entry_results = None
    if USE_AEROCAPTURE and AEROCAPTURE_PHYSICS:
        v_inf_arr_mps = v_inf_arr.to(u.m/u.s).value
        if ENTRY_FPA_SOLVE:
            entry_results = solve_entry_fpa_for_target(v_inf_arr_mps, PERIAPSIS_ALT_M,
                                                       fpa_bounds_deg=FPA_BOUNDS_DEG, tol_m=FPA_SOLVE_TOL_M)
        else:
            entry_results = simulate_entry_pass(v_inf_arr_mps, gamma0_deg=-5.0, rp_target_alt_m=PERIAPSIS_ALT_M)

        print("\n=== Aerocapture Pass (physics) ===")
        print(f"  gamma0:                 {entry_results['gamma0_deg']:.2f} deg")
        print(f"  min altitude:           {entry_results['min_alt_m']/1000:.1f} km")
        print(f"  Δv removed by atmosphere {entry_results['atmo_dv_mps']/1000:.3f} km/s")
        print(f"  max dynamic pressure q: {entry_results['max_q_Pa']/1000:.2f} kPa")
        print(f"  peak heat flux:         {entry_results['max_heat_Wm2']:.0f} W/m^2")
        print(f"  total heat load:        {entry_results['heat_load_Jm2']:.0f} J/m^2")
        print(f"  peak g-load:            {entry_results['max_g']:.2f} g")

        unsafe = []
        if entry_results['max_q_Pa'] > Q_LIMIT_PA:        unsafe.append("q")
        if entry_results['max_heat_Wm2'] > HEAT_LIMIT_WM2: unsafe.append("heat")
        if entry_results['max_g'] > G_LIMIT:               unsafe.append("g-load")
        if unsafe:
            print(f"  [WARN] Aerocapture exceeded limits: {', '.join(unsafe)}")

        dv1_mps = 0.0

    else:
        rp = R_m + PERIAPSIS_ALT_M
        v_inf = v_inf_arr.to(u.m/u.s).value
        v_p   = np.sqrt(v_inf**2 + 2*mu_m/rp)
        v_esc = np.sqrt(2*mu_m/rp)
        dv_atm_min = max(0.0, v_p - v_esc)
        v_peri_target = v_peri_ellip
        dv_atm_to_target = max(0.0, v_p - v_peri_target)
        print("\n=== Aerocapture Requirements (minimal-assumption) ===")
        print(f"Periapsis radius rp:          {rp/1000.0:,.1f} km (alt {PERIAPSIS_ALT_M/1000.0:.1f} km)")
        print(f"Δv_atm_min_to_capture:        {dv_atm_min/1000.0:.3f} km/s")
        print(f"Δv_atm_to_target_ellipse:     {dv_atm_to_target/1000.0:.3f} km/s  (to reach R≈{APOAPSIS_RATIO:.1f})")
        dv1_mps = 0.0  # keeping aerocapture assumption

    dv2_mps = 2.0 * v_apo_ellip * np.sin(theta / 2.0)
    dv3_mps = 0.0 if MINIMUM_ORBIT_MODE else abs(v_circ_m - v_peri_ellip)

    dv1_k = dv1_mps/1000.0; dv2_k = dv2_mps/1000.0; dv3_k = dv3_mps/1000.0
    dv_moi_total = dv1_k + dv2_k + dv3_k

    print(f"\n[MOI Planner] (Aerocapture={'True' if USE_AEROCAPTURE else 'False'} | Physics={'True' if AEROCAPTURE_PHYSICS else 'False'} | MinOrbit={MINIMUM_ORBIT_MODE})")
    print(f"  dv1 (capture at peri):      {dv1_k:.3f} km/s")
    print(f"  dv2 (plane change @ apo):   {dv2_k:.3f} km/s")
    print(f"  dv3 (circularize @ peri):   {dv3_k:.3f} km/s")
    print(f"Total Mars insertion Δv:      {dv_moi_total:.3f} km/s")

    # MONARC for MOI
    m0_moi = m_after_dep
    monarc_prop_full = m_sc_wet - m_sc_dry

    def print_passes(title, passes, prop_start, prop_end):
        print(f"\n[{title}] Perigee-pass burns:")
        for p in passes[:120]:
            prop_now = max(0.0, p['m_end'] - m_sc_dry)
            frac_left = (prop_now / monarc_prop_full * 100.0) if monarc_prop_full>0 else 0.0
            print(f"  pass {p['pass']:02d}: Δv={p['dv_kms']:.4f} km/s, time={p['time_min']:.2f} min, "
                  f"mass {p['m_start']:.1f}→{p['m_end']:.1f} kg (used {p['m_used']:.1f} kg) | "
                  f"MONARC prop left: {prop_now:.1f} kg ({frac_left:.1f}%)")
        if len(passes) > 120:
            print(f"  ... {len(passes)-120} more passes not shown")

    total_monarc_used = monarc_prop_used_dep

    dv1_passes=[]; dv1_deliv=0.0; t1_min=0.0
    m_after1 = m0_moi

    dv2_deliv=0.0; t2_min=0.0
    m_after2 = m_after1
    if dv2_k > 1e-9:
        dt_sec = 0.0
        v_acc  = 0.0
        m      = m_after1
        while v_acc < dv2_k*1000.0 and dt_sec < 3*3600:
            aacc = thrust_monarc / m
            dv = aacc * 1.0
            dm = m * (1.0 - np.exp(-dv / (isp_monarc * g0)))
            m -= dm
            v_acc += dv
            dt_sec += 1.0
        dv2_deliv = min(v_acc/1000.0, dv2_k)
        t2_min = dt_sec/60.0
        m_after2 = m
        used2 = (m_after1 - m_after2); total_monarc_used += used2
        print(f"\n[dv2 @ apo] Δv={dv2_deliv:.3f} km/s, time≈{t2_min:.2f} min, mass {m_after1:.1f}→{m_after2:.1f} kg (used {used2:.1f} kg)")
    else:
        print("\n[dv2 @ apo] Skipped (≈0).")

    dv3_passes=[]; dv3_deliv=0.0; t3_min=0.0
    m_after3 = m_after2
    if dv3_k > 1e-9:
        dv3_passes, dv3_deliv, m_after3, t3_min = perigee_pass_burns_required(
            dv3_k, m_after2, isp_monarc, thrust_monarc, PER_PASS_TIME_MIN_MOI, MAX_PASSES_PER_PHASE, "dv3 circularize"
        )
        print_passes("dv3 (circularize at peri)", dv3_passes, m_after2, m_after3)
        used3 = (m_after2 - m_after3); total_monarc_used += used3
        print("\n[dv3 (circularize) Summary]")
        print(f"  Total Δv:     {dv3_deliv:.3f} km/s")
        print(f"  Total time:   {t3_min:.1f} min")
        print(f"  Prop used:    {used3:.1f} kg")
        print(f"  Passes:       {len(dv3_passes)}")
    else:
        print("\n[dv3 (circularize)] Skipped (Minimum-orbit mode).")

    monarc_prop_remaining = max(0.0, m_after3 - m_sc_dry)
    print("\n--- MONARC Propellant Usage (Total) ---")
    print(f"Prop total (design): {monarc_prop_full:.1f} kg")
    print(f"Prop used (actual):  {total_monarc_used:.1f} kg")
    print(f"Prop remaining:      {monarc_prop_remaining:.1f} kg ({100*monarc_prop_remaining/monarc_prop_full:.1f}% left)")

    monarc_dv_total = _g_dv_monarc_dep_kms + dv1_deliv + dv2_deliv + dv3_deliv
    opt_prop = rocket_eq_prop_for_dv(isp_monarc, m_sc_dry, monarc_dv_total*1000.0)
    opt_wet  = m_sc_dry + opt_prop
    potential_savings = max(0.0, monarc_prop_full - opt_prop)
    print("\n--- Optimal MONARC Prop Estimate ---")
    print(f"Total MONARC Δv delivered: {monarc_dv_total:.3f} km/s")
    print(f"Optimal prop to match Δv:  {opt_prop:.1f} kg  (wet={opt_wet:.1f} kg, dry={m_sc_dry:.1f} kg)")
    print(f"Potential prop savings:    {potential_savings:.1f} kg (vs design {monarc_prop_full:.1f} kg)")

    if MINIMUM_ORBIT_MODE:
        print("\n=== FINAL STATUS: Arrived at Mars ===")
        print(f"Final (captured) ellipse:")
        print(f"  rp = {r_p/1000.0:,.1f} km")
        print(f"  ra = {r_a/1000.0:,.1f} km")
        print(f"  a  = {(0.5*(r_p+r_a))/1000.0:,.1f} km, e = { (r_a-r_p)/(r_a+r_p):.6f}")
        Tm = 2*np.pi*np.sqrt((0.5*(r_p+r_a))**3/mu_m)
        print(f"  Period = {Tm/3600.0:.2f} h")
        final_orbit = ("Mars", r_p, r_a)
    else:
        print("\n=== FINAL STATUS: Arrived at Mars ===")
        print(f"Final (circularized near rp):")
        print(f"  r  = {r_p/1000.0:,.1f} km (e≈0)")
        final_orbit = ("Mars", r_p, r_p)

    initial_capture_orbit = ("Mars", r_p, r_a)

    make_all_plots(depart_dt, arrival_dt, t0, tof_s, r0_km, v_dep, vE_kms, USE_AEROCAPTURE, True,
                   final_orbit=final_orbit,
                   dv_breakdown=(_g_dv_lv2_used_kms, _g_dv_star_used_kms, _g_dv_monarc_dep_kms),
                   entry_results=entry_results,
                   initial_capture_orbit=initial_capture_orbit)

# ===================== PLOTS =====================
def make_all_plots(depart_dt, arrival_dt, t0, tof_s, r0_km, v_dep, vE_kms, use_aerocapture, reached_mars, final_orbit,
                   dv_breakdown=(0.0,0.0,0.0), entry_results=None, initial_capture_orbit=None):
    transfer0 = Orbit.from_vectors(Sun, r0_km, v_dep)
    rE0 = (earth - sun).at(t0).position.km * u.km
    vE0 = (earth - sun).at(t0).velocity.km_per_s * (u.km/u.s)
    rM0 = (mars  - sun).at(t0).position.km * u.km
    vM0 = (mars  - sun).at(t0).velocity.km_per_s * (u.km/u.s)
    earth_orb0 = Orbit.from_vectors(Sun, rE0, vE0)
    mars_orb0  = Orbit.from_vectors(Sun, rM0, vM0)

    N = 400
    tsamp = np.linspace(0.0, tof_s, N) * u.s
    r_sc = np.zeros((N,3)); v_sc = np.zeros((N,3))
    r_ea = np.zeros((N,3)); r_ma = np.zeros((N,3))
    nu_sc = np.zeros(N); speed_sc = np.zeros(N)

    for i, dt_ in enumerate(tsamp):
        o_sc = transfer0.propagate(dt_)
        r_sc[i,:] = o_sc.r.to_value(u.km)
        v_sc[i,:] = o_sc.v.to_value(u.km/u.s)
        nu_sc[i]  = o_sc.nu.to(u.rad).value
        speed_sc[i] = np.linalg.norm(v_sc[i,:])
        r_ea[i,:] = earth_orb0.propagate(dt_).r.to_value(u.km)
        r_ma[i,:] = mars_orb0 .propagate(dt_).r.to_value(u.km)

    d_sc = np.linalg.norm(r_sc, axis=1)
    d_ea = np.linalg.norm(r_ea, axis=1)
    d_ma = np.linalg.norm(r_ma, axis=1)
    t_days = (tsamp.to(u.day)).value

    # 3D heliocentric transfer
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_sc[:,0], r_sc[:,1], r_sc[:,2], label='Spacecraft transfer')
    ax.plot(r_ea[:,0], r_ea[:,1], r_ea[:,2], '--', alpha=0.6, label='Earth arc')
    ax.plot(r_ma[:,0], r_ma[:,1], r_ma[:,2], '--', alpha=0.6, label='Mars arc')
    ax.scatter(r_sc[0,0], r_sc[0,1], r_sc[0,2], s=40, label='Earth @ dep')
    ax.scatter(r_sc[-1,0], r_sc[-1,1], r_sc[-1,2], s=40, label='Mars @ arr')
    ax.scatter(0,0,0, s=80, label='Sun')
    ax.set_title('Heliocentric Transfer (Conic Propagation)')
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)'); ax.set_zlabel('z (km)')
    ax.legend(); plt.tight_layout(); _savefig("heliocentric_3d")

    # 2D XY heliocentric
    plt.figure(figsize=(8,6))
    plt.plot(r_sc[:,0], r_sc[:,1], label='Transfer (conic)')
    plt.plot(r_ea[:,0], r_ea[:,1], '--', alpha=0.6, label='Earth arc')
    plt.plot(r_ma[:,0], r_ma[:,1], '--', alpha=0.6, label='Mars arc')
    plt.scatter(0,0, s=80, label='Sun')
    plt.axis('equal'); plt.grid(True); plt.legend(); plt.title('XY Heliocentric View (Conics)')
    plt.tight_layout(); _savefig("heliocentric_xy")

    # Distance vs time
    plt.figure()
    plt.plot(t_days, d_ea/1e6, label='Earth')
    plt.plot(t_days, d_sc/1e6, label='Spacecraft')
    plt.plot(t_days, d_ma/1e6, label='Mars')
    plt.xlabel('Time since departure (days)'); plt.ylabel('Distance from Sun (10^6 km)')
    plt.title('Heliocentric Distance vs Time')
    plt.grid(True); plt.legend(); plt.tight_layout(); _savefig("heliocentric_distance")

    # True anomaly vs time
    plt.figure()
    nu_deg = np.degrees((nu_sc + 2*np.pi) % (2*np.pi))
    plt.plot(t_days, nu_deg)
    plt.xlabel('Time since departure (days)'); plt.ylabel('True anomaly (deg)')
    plt.title('Spacecraft True Anomaly vs Time')
    plt.grid(True); plt.tight_layout(); _savefig("heliocentric_true_anomaly")

    # Speed vs time
    plt.figure()
    plt.plot(t_days, speed_sc)
    plt.xlabel('Time since departure (days)'); plt.ylabel('Speed (km/s)')
    plt.title('Spacecraft Speed vs Time (Heliocentric)')
    plt.grid(True); plt.tight_layout(); _savefig("heliocentric_speed")

    # Δv breakdown (departure)
    plt.figure()
    bars = ['LV2 leftover', 'STAR', 'MONARC (dep)']
    vals = [dv_breakdown[0], dv_breakdown[1], dv_breakdown[2]]
    plt.bar(bars, vals)
    plt.ylabel('Δv (km/s)'); plt.title('Δv Breakdown — Departure Contributions')
    plt.tight_layout(); _savefig("dv_breakdown")

    # Final orbit plot (Mars or Earth)
    body, rp, ra = final_orbit
    plt.figure()
    if body == "Mars":
        R = R_m
        title = "Final Orbit around Mars"
    else:
        R = R_e
        title = "Final Orbit around Earth"
    a = 0.5*(rp+ra)
    e = (ra - rp)/(ra + rp + 1e-12)
    th = np.linspace(0, 2*np.pi, 400)
    r = (a*(1-e**2)) / (1 + e*np.cos(th))
    x = r*np.cos(th)/1000.0; y = r*np.sin(th)/1000.0
    plt.plot(x, y, label='Final orbit')
    circ = plt.Circle((0,0), R/1000.0, color='orange', alpha=0.3, label=f'{body} radius')
    ax2 = plt.gca(); ax2.add_patch(circ)
    plt.axis('equal'); plt.grid(True); plt.legend(); plt.title(title)
    plt.xlabel('x (km)'); plt.ylabel('y (km)')
    plt.tight_layout(); _savefig("final_orbit")

    # === Initial Capture Orbit plot (orientation-aware, Mars-only) ===
    if PLOT_CAPTURE_3D_ORIENTED and initial_capture_orbit is not None and initial_capture_orbit[0] == "Mars":
        _body_ic, rp_ic, ra_ic = initial_capture_orbit
        R = R_m
        a_ic = 0.5*(rp_ic + ra_ic)
        e_ic = (ra_ic - rp_ic) / (ra_ic + rp_ic + 1e-12)

        # States at arrival
        arr_orb  = transfer0.propagate(tof_s * u.s)
        rSC_arr_km = arr_orb.r.to_value(u.km)
        vSC_arr_kms= arr_orb.v.to_value(u.km/u.s)

        o_mars_arr  = Orbit.from_vectors(Sun, rM0, vM0).propagate(tof_s * u.s)
        rM_arr_km   = o_mars_arr.r.to_value(u.km)
        vM_arr_kms  = o_mars_arr.v.to_value(u.km/u.s)

        v_inf_vec  = (vSC_arr_kms - vM_arr_kms)
        vhat       = v_inf_vec / (np.linalg.norm(v_inf_vec) + 1e-15)

        # Mars orbital normal at arrival
        hM = np.cross(rM_arr_km, vM_arr_kms)
        n_hat = hM / (np.linalg.norm(hM) + 1e-15)

        # Build in-plane basis that contains v∞
        p_hat = np.cross(n_hat, vhat); p_hat /= (np.linalg.norm(p_hat) + 1e-15)
        q_hat = np.cross(vhat, p_hat); q_hat /= (np.linalg.norm(q_hat) + 1e-15)

        th = np.linspace(0.0, 2.0*np.pi, 800)
        r_pf = (a_ic*(1.0 - e_ic**2)) / (1.0 + e_ic*np.cos(th))  # meters

        X = (r_pf * (np.cos(th)*p_hat[0] + np.sin(th)*q_hat[0])) / 1000.0
        Y = (r_pf * (np.cos(th)*p_hat[1] + np.sin(th)*q_hat[1])) / 1000.0
        Z = (r_pf * (np.cos(th)*p_hat[2] + np.sin(th)*q_hat[2])) / 1000.0

        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(7.5,6.0))
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.plot(X, Y, Z, label='Initial capture ellipse')

        # Mars sphere for context
        phi = np.linspace(0, 2*np.pi, 60); the = np.linspace(0, np.pi, 30)
        xs = (R/1000.0) * np.outer(np.cos(phi), np.sin(the))
        ys = (R/1000.0) * np.outer(np.sin(phi), np.sin(the))
        zs = (R/1000.0) * np.outer(np.ones_like(phi), np.cos(the))
        ax3d.plot_surface(xs, ys, zs, alpha=0.25, edgecolor='none', color='orange')

        # Peri/apo markers oriented in space
        r_peri_vec_km = (rp_ic/1000.0) * p_hat
        r_apo_vec_km  = (ra_ic/1000.0) * (-p_hat)
        ax3d.scatter([r_peri_vec_km[0]], [r_peri_vec_km[1]], [r_peri_vec_km[2]], s=24, label='Periapsis')
        ax3d.scatter([r_apo_vec_km[0]],  [r_apo_vec_km[1]],  [r_apo_vec_km[2]],  s=24, label='Apoapsis')

        # v∞ direction arrow at periapsis location
        arr_origin = r_peri_vec_km
        arrow = vhat * (R/1000.0) * 0.6
        ax3d.quiver(arr_origin[0], arr_origin[1], arr_origin[2],
                    arrow[0], arrow[1], arrow[2], length=1.0, normalize=False, label='v∞ dir')

        ax3d.set_title('Initial Capture Orbit around Mars (oriented to v∞)')
        ax3d.set_xlabel('x (km)'); ax3d.set_ylabel('y (km)'); ax3d.set_zlabel('z (km)')
        ax3d.legend(); plt.tight_layout(); _savefig("initial_capture_orbit_mars_oriented")

    # Entry plots
    if entry_results is not None and len(entry_results["time_s"]) > 0:
        tmin = entry_results["time_s"]/60.0
        plt.figure(); plt.plot(tmin, entry_results["alt_m"]/1000.0)
        plt.xlabel('Time (min)'); plt.ylabel('Altitude (km)')
        plt.title('Aerocapture Altitude vs Time'); plt.grid(True); _savefig("entry_altitude")

        plt.figure(); plt.plot(tmin, entry_results["v_mps"]/1000.0)
        plt.xlabel('Time (min)'); plt.ylabel('Speed (km/s)')
        plt.title('Aerocapture Speed vs Time'); plt.grid(True); _savefig("entry_speed")

        plt.figure(); plt.plot(tmin, entry_results["q_Pa"]/1000.0)
        plt.xlabel('Time (min)'); plt.ylabel('Dynamic Pressure q (kPa)')
        plt.title('Dynamic Pressure vs Time'); plt.grid(True); _savefig("entry_q")

        plt.figure(); plt.plot(tmin, entry_results["heat_Wm2"])
        plt.xlabel('Time (min)'); plt.ylabel('Stagnation Heat Flux (W/m²)')
        plt.title('Heat Flux vs Time'); plt.grid(True); _savefig("entry_heat")

        plt.figure(); plt.plot(tmin, entry_results["g_load"])
        plt.xlabel('Time (min)'); plt.ylabel('g-load')
        plt.title('Deceleration vs Time'); plt.grid(True); _savefig("entry_g")

    # === Orientation CSV + status timeline (new) ===
    if SAVE_CSV_ORIENTATION or PLOT_ORIENTATION_TIMELINE:
        import csv, json

        labels = ["Coast"] * N
        labels[0] = "Prograde"
        arr_label = "Retrograde" if (entry_results is None and use_aerocapture is False) else "Aerocapture"
        labels[-1] = arr_label

        csv_rows = []
        for i in range(N):
            Rhat, That, Nhat = _rtn_basis(r_sc[i,:], v_sc[i,:])
            if labels[i] == "Prograde":
                orient_rtn = (0.0, 1.0, 0.0)
            elif labels[i] == "Retrograde":
                orient_rtn = (0.0, -1.0, 0.0)
            elif labels[i] == "Radial-out":
                orient_rtn = (1.0, 0.0, 0.0)
            elif labels[i] == "Radial-in":
                orient_rtn = (-1.0, 0.0, 0.0)
            elif labels[i] == "Normal +":
                orient_rtn = (0.0, 0.0, 1.0)
            elif labels[i] == "Normal -":
                orient_rtn = (0.0, 0.0, -1.0)
            elif labels[i] == "Aerocapture":
                orient_rtn = (np.nan, np.nan, np.nan)
            else:
                orient_rtn = (np.nan, np.nan, np.nan)

            csv_rows.append({
                "t_days":      t_days[i],
                "r_x_km":      r_sc[i,0], "r_y_km": r_sc[i,1], "r_z_km": r_sc[i,2],
                "v_x_kmps":    v_sc[i,0], "v_y_kmps": v_sc[i,1], "v_z_kmps": v_sc[i,2],
                "Rhat_x": Rhat[0], "Rhat_y": Rhat[1], "Rhat_z": Rhat[2],
                "That_x": That[0], "That_y": That[1], "That_z": That[2],
                "Nhat_x": Nhat[0], "Nhat_y": Nhat[1], "Nhat_z": Nhat[2],
                "orient_rtn_R": orient_rtn[0], "orient_rtn_T": orient_rtn[1], "orient_rtn_N": orient_rtn[2],
                "orient_label": labels[i]
            })

        if SAVE_CSV_ORIENTATION:
            try:
                with open("timeline_orient.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                    w.writeheader(); w.writerows(csv_rows)
                print("[CSV] wrote timeline_orient.csv")
            except Exception as e:
                print("[CSV warning] timeline_orient.csv:", e)

        if SAVE_JSON_ORIENTATION:
            try:
                summary = {
                    "dep_label": labels[0],
                    "arr_label": labels[-1],
                    "N_samples": N,
                    "notes": "orient_rtn_* are unit-vector components in RTN for the recommended axis; NaN = coast/EDL."
                }
                with open("timeline_orient_summary.json", "w") as jf:
                    json.dump(summary, jf, indent=2)
                print("[JSON] wrote timeline_orient_summary.json")
            except Exception as e:
                print("[JSON warning] timeline_orient_summary.json:", e)

        if PLOT_ORIENTATION_TIMELINE:
            legend_keys = ["Coast","Prograde","Retrograde","Radial-out","Radial-in","Normal +","Normal -","Aerocapture"]
            key_to_int = {k:i for i,k in enumerate(legend_keys)}
            y = np.array([key_to_int.get(lbl, 0) for lbl in labels], dtype=float)

            plt.figure(figsize=(9,3))
            plt.step(t_days, y, where='post')
            plt.yticks(range(len(legend_keys)), legend_keys)
            plt.xlabel('Time since departure (days)')
            plt.title('Spacecraft Orientation Status vs Time (RTN label)')
            plt.grid(True, axis='x', alpha=0.4)
            plt.tight_layout(); _savefig("orientation_timeline")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')

if __name__ == "__main__":
    main()
