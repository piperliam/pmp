import math, sys, os, random, csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
import trimesh

# =============================================================================
# ============================== USER TOGGLES =================================
# =============================================================================

# --- Files & units ---
STL_PATH     = "aeroshell_v4.5.stl"      # STL filename (same folder)
UNITS        = "mm"             # m | mm | cm | in | ft
SCALE        = 1.0              # extra scale factor applied after UNITS

# --- Centering (after orientation) ---
CENTER       = "mass"           # none | mass | centroid | bbox

# --- CAD orientation controls (make CAD "down is down") ---
CAD_UP       = "Z+"             # Z+ | Z- | Y+ | Y- | X+ | X-
CAD_YAW_DEG  = 0.0              # spin around CAD-up (deg), after aligning to +Z

# --- Flow controls (define the 3-D wind direction v̂) ---
FLOW_MODE    = "CUSTOM"         # ALONG_UP | CUSTOM
FLOW_SENSE   = "up"             # up | down  (for ALONG_UP → v̂ = ±Z)
FLOW         = "+z"             # (CUSTOM) "+x", "-y,+z", etc.
YAW = PITCH = ROLL = 0.0        # (CUSTOM) extra Euler, deg

# --- Aerodynamic model (pressure) ---
METHOD       = "newtonian"      # newtonian | modified
GAMMA        = 1.4
MACH         = 12.0
T_INF        = 220.0            # K
P_INF        = 300.0            # Pa
R_AIR        = 287.0            # J/kg/K

# --- Friction (optional additive CD_f) ---
INCLUDE_FRICTION = True
CF_MODEL    = "turbulent"       # laminar | turbulent | schlichting

# --- Thermal proxy (Sutton–Graves & adiabatic wall temp) ---
THERMAL     = True
NOSE_RADIUS = -1.0              # if <=0 → estimate from windward cap
PRANDTL     = 0.72
RECOVERY    = "turbulent"       # turbulent | laminar (for Taw recovery factor)

# --- Plot control (global) ---
SHOW_PLOTS  = True              # show figure windows?
SAVE_PLOTS  = True             # save PNGs?
SAVE_PREFIX = "aeroshell_run_4_friction"            # filename prefix when saving

# --- Overview render (3D + 2D panel with CFD Mach overlay) ---
PLOT_OVERVIEW         = True
LOCK_2D_TO_GLOBAL     = False     # 2D panel axes in global frame vs flow-based
INVERT_2D_Y           = False
ARROW2D               = "vertical"  # vertical | horizontal
SILHOUETTE            = "both"      # convex | true | both
SIL_RES               = 1000

# --- Projection plane for 2-D CFD + 2-D panel ---
# "XY" = frontal; "XZ" = side X–Z; "YZ" = side Y–Z
PROJECTION_PLANE      = "XZ"

# --- 3-D Cp renders at multiple views ---
PLOT_3D_ANGLES        = True
VIEWS_3D = [
    ("iso_plus",   25,  35),
    ("iso_minus",  25, -145),
    ("side_x+",     5,   0),
    ("side_x-",     5, 180),
    ("side_y+",     5,  90),
    ("side_y-",     5, -90),
    ("top",        90,  90),   # +Y vertical in frame
    ("front",       0,  90),
]

# --- Thermal proxy renders (colored by q̇ proxy) ---
PLOT_3D_THERMAL       = True
THERMAL_VIEWS_3D = [
    ("thermal_iso+", 25, 35),
    ("thermal_top",  90, 90),
]

# --- 3-D flow lines (smoke) ---
PLOT_3D_FLOWLINES = False
FLOWLINES_MODE    = "cfd"     # "cfd" (curved from 2D field) | "straight" (guide lines)
FLOWLINES_DT      = 0.35      # RK2 step fraction of min(dx,dy)
FLOWLINES_STEPS   = 2000
FLOWLINES_RAKE_XZ = (22, 7)   # number of seeds along X, Z
FLOWLINES_LEN     = 2.6       # only for "straight" mode
FLOWLINES_OFFSET  = 1.2       # upstream offset in body widths
FLOWLINES_JITTER  = 0.10      # 0…0.3 good

FLOWLINES_VIEWS_3D = [
    ("smoke_iso+",  25,  35),
    ("smoke_side",   5,   0),   # +Y vertical
    ("smoke_front",  0,  90),
    ("smoke_top",   90,  90),   # top view with +Y up
]

# --- 2-D CFD settings (Euler/HLLE) ---
ENABLE_CFD          = True
CFD_NX, CFD_NY      = 800, 600      # higher res → thinner shock
CFD_BOX_EXPAND      = (0.7, 1.3)    # (left/right, down/up) – tighter box
CFL                 = 0.40
STEPS               = 3000
RES_TOL             = 2e-6
REPORT_EVERY        = 100
POS_RHO_FLOOR       = 1e-5
POS_P_FLOOR         = 1e-6

# Rotate CFD fields for display so +Y is vertical upward (bottom → top inflow)
CFD_ROTATE_VIEW     = 90      # IMPORTANT: makes all maps look bottom→top

# Extra CFD field plots (single overview rotation)
PLOT_CFD_FIELDS     = True
CFD_FIELD_KEYS      = ["mach", "pressure", "density", "speed", "gradrho"]
STREAMLINES_ON_FIELDS = True
STREAMLINE_DENSITY  = 1.8     # bump this for more 2-D streamlines

# Multi-view pressure maps (display rotations)
PLOT_CFD_PRESSURE_VIEWS = True
CFD_PRESSURE_ROTATIONS  = [90]

# Centerline chart
PLOT_CENTERLINE     = True

# Colored PLY export (Cp on faces), "" disables
EXPORT_PLY          = ""        # e.g. "flow_cp.ply"

# --- AoA sweep (fast Newtonian/modified; CSV output & plot) ---
DO_ALPHA_SWEEP   = False
ALPHAS_DEG       = [-10, -5, -2, 0, 2, 5, 10]
L_REF_MOMENT     = 1.0
SAVE_SWEEP_CSV   = True
SWEEP_CSV_NAME   = f"{SAVE_PREFIX}_sweep.csv"
PLOT_SWEEP       = True

# =============================================================================
# ============================== IMPLEMENTATION ===============================
# =============================================================================

UNIT_SCALE = {"m":1.0,"mm":1e-3,"cm":1e-2,"in":0.0254,"ft":0.3048}
random.seed(3)
np.set_printoptions(precision=4, suppress=True)

def unit(v): n=np.linalg.norm(v); return v/(n if n>0 else 1.0)

def parse_flow(s):
    v = np.zeros(3)
    if not s: return np.array([1.0,0,0])
    for tok in [t for t in s.replace(' ','').split(',') if t]:
        sgn = -1.0 if tok[0]=='-' else +1.0
        ax  = tok[1:].lower() if tok[0] in '+-' else tok.lower()
        if   ax=='x': v[0]+=sgn
        elif ax=='y': v[1]+=sgn
        elif ax=='z': v[2]+=sgn
    return unit(v) if np.linalg.norm(v)>0 else np.array([1.0,0,0])

def rot_xyz(v, yaw=0.0, pitch=0.0, roll=0.0):
    vz, vy, vx = map(math.radians, (yaw, pitch, roll))
    cz,sz = math.cos(vz), math.sin(vz)
    cy,sy = math.cos(vy), math.sin(vy)
    cx,sx = math.cos(vx), math.sin(vx)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    return (Rz @ (Ry @ (Rx @ v)))

def parse_axis(ax):
    ax = ax.strip().upper()
    sign = +1.0 if ax.endswith('+') else -1.0
    base = ax[0]
    basev = {'X':np.array([1,0,0]), 'Y':np.array([0,1,0]), 'Z':np.array([0,0,1])}[base]
    return sign*basev

def rotation_matrix_from_vectors(a, b):
    a = unit(a); b = unit(b)
    v = np.cross(a,b); c = float(np.dot(a,b)); s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0: return np.eye(3)
        t = np.array([1,0,0]) if abs(a[0])<0.9 else np.array([0,1,0])
        v = unit(np.cross(a,t))
        K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        return np.eye(3) + 2*K@K
    K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + K + K@K*((1-c)/(s*s))

def euler_matrix_z(yaw_deg):
    z = math.radians(yaw_deg); cz,sz = math.cos(z), math.sin(z)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]); T = np.eye(4); T[:3,:3]=Rz; return T

def ortho_basis_from(v):
    v = unit(v)
    t = np.array([0,0,1.0]) if abs(v[2])<0.9 else np.array([0,1.0,0])
    u = unit(np.cross(t, v)); w = np.cross(v, u)
    return u, w, v

def recenter(mesh, mode):
    if mode=="none": return
    t = mesh.center_mass if mode=="mass" else (mesh.centroid if mode=="centroid" else (mesh.bounds[0]+mesh.bounds[1])/2.0)
    mesh.apply_translation(-t)

def projected_area(mesh, vhat):
    n = mesh.face_normals; A = mesh.area_faces
    cos_up = np.maximum(0.0, -n @ vhat)
    return float(np.sum(A * cos_up))

def cp_max_modified_newtonian(gamma, M):
    g = gamma; M1 = M
    p2_p1 = 1.0 + 2.0*g/(g+1.0)*(M1*M1 - 1.0)
    M2_sq = (1.0 + 0.5*(g-1.0)*M1*M1)/(g*M1*M1 - 0.5*(g-1.0))
    p0_2_p2 = (1.0 + 0.5*(g-1.0)*M2_sq)**(g/(g-1.0))
    p0_2_p1 = p2_p1 * p0_2_p2
    q_inf_over_p1 = 0.5*g*M1*M1
    return (p0_2_p1 - 1.0) / q_inf_over_p1

def newtonian_coeffs(mesh, vhat, method="newtonian", gamma=1.4, M=8.0):
    try: mesh.fix_normals()
    except Exception: pass
    n = mesh.face_normals; A = mesh.area_faces
    c = np.maximum(0.0, -n @ vhat)   # windward cosine
    Aref = projected_area(mesh, vhat)
    if Aref <= 0: raise RuntimeError("Projected area is zero for this flow direction.")
    if method == "modified":
        Cp = cp_max_modified_newtonian(gamma, M)*(c**2)
    else:
        Cp = 2.0*(c**2)
    Cvec_p = -((Cp[:,None]*n)*A[:,None]).sum(axis=0)/Aref
    Cd_p = float(Cvec_p @ vhat)
    tri_cent = mesh.triangles_center
    w = Cp*A; W = np.sum(w) if np.sum(w)>0 else 1.0
    r_cp = (tri_cent.T @ w)/W
    return Cd_p, Cvec_p, Aref, r_cp, Cp

def safe_volume(mesh):
    if mesh.is_volume and mesh.volume>0: return float(mesh.volume), "mesh"
    try:
        vh = mesh.convex_hull
        if vh.is_volume and vh.volume>0: return float(vh.volume), "convex_hull"
    except Exception:
        pass
    try:
        ext = mesh.bounds[1]-mesh.bounds[0]; pitch = max(min(ext)/256.0, 1e-6)
        vg = mesh.voxelized(pitch)
        try: vg = vg.filled()
        except AttributeError:
            try: vg = vg.fill()
            except Exception: pass
        m = getattr(vg,'matrix',None)
        if m is not None:
            vol = float(np.count_nonzero(m)*(vg.pitch**3))
            if vol>0: return vol, "voxel"
    except Exception:
        pass
    return float('nan'), "none"

def convex_hull_2d(points):
    pts = np.unique(np.round(points, 12), axis=0)
    if pts.shape[0] <= 3: return pts
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]; upper=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p) <= 0: lower.pop()
        lower.append(tuple(p))
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p) <= 0: upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1]+upper[:-1], dtype=float)

def true_silhouette_mask(P, F, res=700, margin=0.05):
    mins = P.min(axis=0); maxs = P.max(axis=0)
    L = float(np.max(maxs - mins))
    mins = mins - margin*L; maxs = maxs + margin*L
    xs = np.linspace(mins[0], maxs[0], res, endpoint=True)
    ys = np.linspace(mins[1], maxs[1], res, endpoint=True)
    Xc, Yc = np.meshgrid(xs, ys, indexing='xy')
    mask = np.zeros((res, res), dtype=bool)

    def idx_range(vmin, vmax, grid):
        i0 = max(0, int(np.floor((vmin - grid[0]) / (grid[1]-grid[0]))))
        i1 = min(len(grid)-1, int(np.ceil((vmax - grid[0]) / (grid[1]-grid[0]))))
        if i1 < i0: i0, i1 = i1, i0
        return i0, i1

    for tri in P[F]:
        xmin, ymin = tri.min(axis=0); xmax, ymax = tri.max(axis=0)
        ix0, ix1 = idx_range(xmin, xmax, xs)
        iy0, iy1 = idx_range(ymin, ymax, ys)
        if ix1<ix0 or iy1<iy0: continue
        subX = Xc[iy0:iy1+1, ix0:ix1+1]
        subY = Yc[iy0:iy1+1, ix0:ix1+1]
        pts  = np.column_stack([subX.ravel(), subY.ravel()])
        inside = MplPath(tri).contains_points(pts).reshape(subX.shape)
        mask[iy0:iy1+1, ix0:ix1+1] |= inside

    area = float(np.count_nonzero(mask)) * (xs[1]-xs[0]) * (ys[1]-ys[0])
    return mask, xs, ys, area

def sutherland_mu(T, mu0=1.716e-5, T0=273.15, S=110.4):
    return mu0*(T/T0)**1.5*(T0+S)/(T+S)

def friction_cd(S_wet, Aref, L_char, rho, V, Mach, model="turbulent"):
    if Aref<=0 or S_wet<=0 or L_char<=0: return 0.0, 0.0
    mu = sutherland_mu(T=T_INF); Re = max(rho*V*L_char/mu, 1e3)
    if model=="laminar": Cf = 1.328/math.sqrt(Re)
    elif model=="schlichting": Cf = 0.455/(math.log10(Re)**2.58)
    else: Cf = 0.074/(Re**0.2)
    Cf /= math.sqrt(1.0 + 0.2*Mach*Mach)  # compressibility correction
    return Cf*(S_wet/Aref), Cf

def estimate_nose_radius(mesh, vhat, cap_frac=0.02):
    V = mesh.vertices; s = V @ vhat
    smin, smax = float(np.min(s)), float(np.max(s))
    if smax <= smin: return float('nan')
    thresh = smin + cap_frac*(smax - smin)
    mask = s <= thresh
    if np.count_nonzero(mask) < 10: return float('nan')
    u,w,_ = ortho_basis_from(vhat)
    P = np.column_stack([V[mask] @ u, V[mask] @ w])
    pts = np.unique(np.round(P,12), axis=0)
    if pts.shape[0] < 3: return float('nan')
    hull = convex_hull_2d(pts)
    if hull.shape[0] < 3: return float('nan')
    x,y = hull[:,0], hull[:,1]
    Acap = 0.5*abs(np.dot(x,np.roll(y,-1)) - np.dot(y,np.roll(x,-1)))
    return math.sqrt(Acap/math.pi)

def sutton_graves_qdot_Wm2(rho, V, Rn):
    if not (Rn>0): return float('nan')
    q_cm2 = 1.83e-4 * math.sqrt(rho / Rn) * (V**3)  # W/cm^2
    return q_cm2 * 1.0e4  # → W/m^2

def thermal_face_proxy(mesh, vhat, q_stag):
    n = mesh.face_normals
    cosw = np.clip(-(n @ vhat), 0.0, 1.0)
    return q_stag * (cosw**1.0)

# === CFD core: 2-D Euler with HLLE ===
def cons2prim(U, gamma):
    rho = U[...,0]
    u   = U[...,1]/np.maximum(rho, POS_RHO_FLOOR)
    v   = U[...,2]/np.maximum(rho, POS_RHO_FLOOR)
    E   = U[...,3]
    p   = (gamma-1.0)*(E - 0.5*rho*(u*u + v*v))
    p   = np.maximum(p, POS_P_FLOOR)
    a   = np.sqrt(np.maximum(gamma*p/np.maximum(rho, POS_RHO_FLOOR), 1e-12))
    return rho, u, v, p, a

def prim2cons(rho, u, v, p, gamma):
    E = p/(gamma-1.0) + 0.5*rho*(u*u + v*v)
    return np.stack([rho, rho*u, rho*v, E], axis=-1)

def flux_x(U, gamma):
    rho,u,v,p,_ = cons2prim(U, gamma); H = (U[...,3] + p)/rho
    return np.stack([rho*u, rho*u*u + p, rho*u*v, rho*u*H], axis=-1)

def flux_y(U, gamma):
    rho,u,v,p,_ = cons2prim(U, gamma); H = (U[...,3] + p)/rho
    return np.stack([rho*v, rho*u*v, rho*v*v + p, rho*v*H], axis=-1)

def hlle(U_L, U_R, gamma, axis='x'):
    rhoL,uL,vL,pL,aL = cons2prim(U_L, gamma)
    rhoR,uR,vR,pR,aR = cons2prim(U_R, gamma)
    if axis=='x':
        vnL, vnR = uL, uR; FL, FR = flux_x(U_L, gamma), flux_x(U_R, gamma)
    else:
        vnL, vnR = vL, vR; FL, FR = flux_y(U_L, gamma), flux_y(U_R, gamma)
    SL = np.minimum(vnL - aL, vnR - aR); SR = np.maximum(vnL + aL, vnR + aR)
    denom = np.where(SR-SL > 1e-12, 1.0/(SR-SL), 0.0)
    F = (SR[...,None]*FL - SL[...,None]*FR + SL[...,None]*SR[...,None]*(U_R - U_L)) * denom[...,None]
    F = np.where((SL>=0)[...,None], FL, F)
    F = np.where((SR<=0)[...,None], FR, F)
    return F

# ---------- MUSCL reconstruction (2nd order) ----------
def minmod(a, b):
    """TVD minmod limiter."""
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

def reconstruct_x(U):
    """
    MUSCL reconstruction in x-direction.
    U: [ny, nx, nvar] cell-centered → UL, UR at interfaces [ny, nx-1, nvar]
    UL = state to LEFT of interface, UR = state to RIGHT.
    """
    slope = np.zeros_like(U)
    dL = U - np.roll(U, 1, axis=1)        # U_i - U_{i-1}
    dR = np.roll(U, -1, axis=1) - U       # U_{i+1} - U_i
    slope[:, 1:-1, :] = minmod(dL[:, 1:-1, :], dR[:, 1:-1, :])

    UL = U[:, :-1, :] + 0.5 * slope[:, :-1, :]
    UR = U[:, 1:,  :] - 0.5 * slope[:, 1:,  :]
    return UL, UR

def reconstruct_y(U):
    """
    MUSCL reconstruction in y-direction.
    U: [ny, nx, nvar] cell-centered → UD, UU at interfaces [ny-1, nx, nvar]
    UD = state BELOW interface, UU = state ABOVE.
    """
    slope = np.zeros_like(U)
    dL = U - np.roll(U, 1, axis=0)        # U_j - U_{j-1}
    dR = np.roll(U, -1, axis=0) - U       # U_{j+1} - U_j
    slope[1:-1, :, :] = minmod(dL[1:-1, :, :], dR[1:-1, :, :])

    UD = U[:-1, :, :] + 0.5 * slope[:-1, :, :]
    UU = U[1:,  :, :] - 0.5 * slope[1:,  :, :]
    return UD, UU
# ------------------------------------------------------

def mirror_state(U, gamma, axis='+x'):
    rho,u,v,p,a = cons2prim(U, gamma)
    if axis in ('+x','-x'): u = -u
    else:                   v = -v
    return prim2cons(rho,u,v,p,gamma)

def run_cfd_on_silhouette(hull, bounds, gamma, Mach,
                          nx, ny, box_expand_lr=1.2, box_expand_ud=2.1,
                          steps=4000, cfl=0.45, tol=1e-5, report_every=500):
    # Inflow is +Y (bottom→top before display rotation)
    rho_inf = 1.0; U_inf = Mach; p_inf = 1.0/gamma
    u_inf, v_inf = (0.0, U_inf)

    mins = bounds[0]; maxs = bounds[1]
    Lx, Ly = (maxs - mins)
    bodyW  = max(Lx, Ly)
    x0 = mins[0] - box_expand_lr*bodyW
    x1 = maxs[0] + box_expand_lr*bodyW
    y0 = mins[1] - box_expand_ud*bodyW
    y1 = maxs[1] + 0.75*box_expand_ud*bodyW

    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    dx = xs[1]-xs[0]; dy = ys[1]-ys[0]
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    path = MplPath(hull)
    solid = path.contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape((ny,nx))

    U = np.zeros((ny,nx,4), dtype=float)
    U[...,0] = rho_inf
    U[...,1] = rho_inf*u_inf
    U[...,2] = rho_inf*v_inf
    U[...,3] = p_inf/(gamma-1.0) + 0.5*rho_inf*(u_inf*u_inf + v_inf*v_inf)

    Uprev = U.copy()
    for it in range(1, steps+1):
        rho,u,v,p,a = cons2prim(U, gamma)
        Smax = np.max(np.abs(u)+a); Tmax = np.max(np.abs(v)+a)
        dt = cfl * min(dx/max(Smax,1e-12), dy/max(Tmax,1e-12))

        # ---------- X-direction (2nd-order MUSCL + HLLE) ----------
        UL, UR = reconstruct_x(U)
        Fx = hlle(UL, UR, gamma, axis='x')
        solid_L, solid_R = solid[:, :-1], solid[:, 1:]
        mask = (~solid_L) & ( solid_R)
        if np.any(mask):
            Fx[mask] = hlle(UL[mask], mirror_state(UL[mask], gamma, '+x'), gamma, 'x')
        mask = ( solid_L) & (~solid_R)
        if np.any(mask):
            Fx[mask] = hlle(mirror_state(UR[mask], gamma, '-x'), UR[mask], gamma, 'x')

        # ---------- Y-direction (2nd-order MUSCL + HLLE) ----------
        UD, UU = reconstruct_y(U)
        Fy = hlle(UD, UU, gamma, axis='y')
        solid_D, solid_U = solid[:-1, :], solid[1:, :]
        mask = (~solid_D) & ( solid_U)
        if np.any(mask):
            Fy[mask] = hlle(UD[mask], mirror_state(UD[mask], gamma, '+y'), gamma, 'y')
        mask = ( solid_D) & (~solid_U)
        if np.any(mask):
            Fy[mask] = hlle(mirror_state(UU[mask], gamma, '-y'), UU[mask], gamma, 'y')

        # ---------- Update ----------
        U[:,1:-1,:] -= dt/dx * (Fx[:,1:,:] - Fx[:,:-1,:])
        U[1:-1,:, :] -= dt/dy * (Fy[1:,:,:] - Fy[:-1,:,:])

        # Enforce solid interior
        U[solid] = Uprev[solid]

        # Inflow/outflow & sides
        U[0,:,:] = U[1,:,:]
        U[0,:,0]=1.0; U[0,:,1]=rho_inf*u_inf; U[0,:,2]=rho_inf*v_inf
        U[0,:,3]=p_inf/(GAMMA-1.0)+0.5*rho_inf*(u_inf*u_inf+v_inf*v_inf)
        U[-1,:,:] = U[-2,:,:]
        U[:,0,:]  = U[:,1,:]
        U[:,-1,:] = U[:,-2,:]

        # Positivity
        rho,u,v,p,a = cons2prim(U, gamma)
        rho = np.maximum(rho, POS_RHO_FLOOR); p = np.maximum(p, POS_P_FLOOR)
        U = prim2cons(rho,u,v,p,gamma)

        res = np.linalg.norm((U[1:-1,1:-1,:]-Uprev[1:-1,1:-1,:]).reshape(-1,4))
        ref = np.linalg.norm(Uprev[1:-1,1:-1,:].reshape(-1,4)) + 1e-12
        if (it % REPORT_EVERY) == 0 or it == 1:
            umax = float(np.nanmax(np.sqrt(u*u+v*v)))
            pmin, pmax = float(np.nanmin(p)), float(np.nanmax(p))
            print(f"  [CFD] step {it:5d} | dt={dt:.3e} | res={res/ref:.3e} | "
                  f"|V|max={umax:.2f} | p[min,max]=[{pmin:.2g},{pmax:.2g}]")
        if res/ref < tol:
            print(f"  [CFD] converged at step {it} (res={res/ref:.3e})")
            break
        Uprev[...] = U
    return xs, ys, solid, U

def cfd_summary_report(xs, ys, solid, U, Deq, gamma):
    """
    Print a short summary of the 2-D CFD:
      - Mach range outside the body
      - subsonic area fraction (M<1)
      - bow-shock stand-off along centerline, using |∇ρ| peak
    Returns a dict with indices/coordinates for plotting.
    """
    rho, u, v, p, a = cons2prim(U, gamma)
    speed = np.sqrt(np.maximum(u*u + v*v, 0.0))
    M = speed / np.maximum(a, 1e-12)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    fluid_mask = ~solid
    M_field = M[fluid_mask]

    if M_field.size > 0:
        M_inf_num = float(np.percentile(M_field, 95))
        M_min     = float(np.min(M_field))
        M_mean    = float(np.mean(M_field))
    else:
        M_inf_num = M_min = M_mean = float('nan')

    drdx = (np.roll(rho, -1, axis=1) - np.roll(rho, 1, axis=1)) / (2*dx)
    drdy = (np.roll(rho, -1, axis=0) - np.roll(rho, 1, axis=0)) / (2*dy)
    gr   = np.sqrt(drdx*drdx + drdy*drdy)

    xmid = 0.5*(xs[0] + xs[-1])
    ix = int(np.clip(round((xmid - xs[0]) / dx), 0, len(xs)-1))

    solid_col = solid[:, ix]
    nose_idx = np.where(solid_col)[0]
    if nose_idx.size > 0:
        j_nose = int(nose_idx[0])
        y_nose = ys[j_nose]
    else:
        j_nose = None
        y_nose = float('nan')

    if j_nose is not None and j_nose > 1:
        gr_col = gr[:j_nose, ix]
        j_shock = int(np.argmax(gr_col))
        y_shock = ys[j_shock]
        standoff = y_nose - y_shock
    else:
        j_shock = None
        y_shock = float('nan')
        standoff = float('nan')

    area_tot = float(np.count_nonzero(fluid_mask)) * dx * dy
    area_sub = float(np.count_nonzero(fluid_mask & (M < 1.0))) * dx * dy
    frac_sub = area_sub / area_tot if area_tot > 0 else float('nan')

    print("\n[CFD summary (2-D Euler)]")
    if M_field.size > 0:
        print(f"  Mach (outside body):  min={M_min:.3f}, mean={M_mean:.3f}, "
              f"95th pct≈{M_inf_num:.3f}")
    print(f"  Subsonic area fraction (M<1): {frac_sub:.3f}")
    if j_nose is not None and j_shock is not None:
        Deq_safe = Deq if (Deq is not None and Deq > 0) else float('nan')
        print(f"  Bow-shock stand-off (centerline): {standoff:.4g} m"
              f"   (~{standoff/Deq_safe:.3f} × D_eq)")
        print(f"    centerline x index={ix}, y_shock={y_shock:.4g} m, "
              f"y_nose={y_nose:.4g} m")
    else:
        print("  Bow shock / nose centerline locations could not be robustly determined.")

    return {"ix": ix, "y_shock": y_shock, "y_nose": y_nose}

# === helpers for CFD visualization & streamlines ===

def orient_xy_for_plot(xs, ys, A, solid, rotate_deg=0):
    """
    Rotate (A, solid) by multiples of 90 deg for display, and ALWAYS return
    x- and y-axes that are strictly increasing (required by streamplot).
    """
    k = (int(rotate_deg) // 90) % 4
    if k == 0:
        xs_out, ys_out = xs, ys
        A_out, S_out = A, solid
    elif k == 1:  # 90° CCW
        xs_out, ys_out = ys, xs
        A_out, S_out = np.rot90(A, 1), np.rot90(solid, 1)
    elif k == 2:  # 180°
        xs_out, ys_out = xs, ys
        A_out, S_out = np.rot90(A, 2), np.rot90(solid, 2)
    else:         # k == 3, 270° CCW
        xs_out, ys_out = ys, xs
        A_out, S_out = np.rot90(A, 3), np.rot90(solid, 3)
    return xs_out, ys_out, A_out, S_out

def orient_vector_field(xs, ys, u, v, solid, rotate_deg=0):
    """
    Rotate a 2-D vector field (u,v) and the solid mask by multiples of 90°
    so we can plot with +y up. Handles both array rotation and component
    transformation. For the 90° case we flip sign so far-field arrows
    point towards the body in the Mach plot.
    """
    k = (int(rotate_deg) // 90) % 4

    if k == 0:   # no rotation
        return xs, ys, u, v, solid

    elif k == 1:  # 90° CCW
        xs_out, ys_out = ys, xs
        u_r = np.rot90(u, 1)
        v_r = np.rot90(v, 1)
        solid_r = np.rot90(solid, 1)
        # 90° CCW rotation of the vector:
        # [u'; v'] = [[0,-1],[1,0]] [u; v]
        u_out = -v_r
        v_out =  u_r
        # Flip so that freestream (original +v) becomes left→right on the plot
        u_out = -u_out
        v_out = -v_out
        return xs_out, ys_out, u_out, v_out, solid_r

    elif k == 2:  # 180°
        xs_out, ys_out = xs, ys
        u_r = np.rot90(u, 2)
        v_r = np.rot90(v, 2)
        solid_r = np.rot90(solid, 2)
        u_out = -u_r
        v_out = -v_r
        return xs_out, ys_out, u_out, v_out, solid_r

    else:         # k == 3, 270° CCW (90° CW)
        xs_out, ys_out = ys, xs
        u_r = np.rot90(u, 3)
        v_r = np.rot90(v, 3)
        solid_r = np.rot90(solid, 3)
        # 270° CCW: [u'; v'] = [v; -u]
        u_out =  v_r
        v_out = -u_r
        return xs_out, ys_out, u_out, v_out, solid_r

def _bilinear(A, xs, ys, x, y):
    i = np.searchsorted(xs, x) - 1
    j = np.searchsorted(ys, y) - 1
    if i < 0 or j < 0 or i >= len(xs)-1 or j >= len(ys)-1:
        return np.nan
    tx = (x - xs[i])/(xs[i+1]-xs[i]); ty = (y - ys[j])/(ys[j+1]-ys[j])
    return ((1-tx)*(1-ty)*A[j,i] + tx*(1-ty)*A[j,i+1] +
            (1-tx)*ty*A[j+1,i] + tx*ty*A[j+1,i+1])

def integrate_streamline(xs, ys, u, v, solid, x0, y0, dt, nsteps):
    pts = []
    x, y = float(x0), float(y0)
    for _ in range(nsteps):
        if (x < xs[0]) or (x > xs[-1]) or (y < ys[0]) or (y > ys[-1]): break
        i = np.searchsorted(xs, x) - 1; j = np.searchsorted(ys, y) - 1
        if i>=0 and j>=0 and i<len(xs)-1 and j<len(ys)-1 and solid[j,i]: break
        ux1 = _bilinear(u, xs, ys, x, y); vy1 = _bilinear(v, xs, ys, x, y)
        if not np.isfinite(ux1) or not np.isfinite(vy1): break
        xm, ym = x + 0.5*dt*ux1, y + 0.5*dt*vy1
        ux2 = _bilinear(u, xs, ys, xm, ym); vy2 = _bilinear(v, xs, ys, xm, ym)
        if not np.isfinite(ux2) or not np.isfinite(vy2): break
        x += dt*ux2; y += dt*vy2
        pts.append((x, y))
    return np.array(pts) if len(pts) else None

def build_flowlines_from_cfd_xy(mesh, xs, ys, u, v, solid,
                                rake_xz=(18,6), dt_frac=0.35, steps=900,
                                upstream=1.2, jitter=0.10):
    V = mesh.vertices
    xmin,xmax = float(np.min(V[:,0])), float(np.max(V[:,0]))
    zmin,zmax = float(np.min(V[:,2])), float(np.max(V[:,2]))
    Dx = xmax-xmin; Dz = zmax-zmin
    nx, nz = rake_xz
    xs_seeds = np.linspace(xmin-0.15*Dx, xmax+0.15*Dx, nx)
    zs_seeds = np.linspace(zmin-0.15*Dz, zmax+0.15*Dz, nz)
    jx = (np.random.rand(nx*nz)-0.5)*jitter*Dx
    jz = (np.random.rand(nx*nz)-0.5)*jitter*Dz

    # start slightly upstream in Y (bottom of domain)
    ystart = ys[0] + 0.02*(ys[-1]-ys[0])
    dt = dt_frac * min(xs[1]-xs[0], ys[1]-ys[0])

    lines = []
    k = 0
    for xi in xs_seeds:
        for zi in zs_seeds:
            x0 = xi + jx[k]; z0 = zi + jz[k]
            path2d = integrate_streamline(xs, ys, u, v, solid, x0, ystart, dt, steps)
            if path2d is not None and path2d.shape[0] > 3:
                X = path2d[:,0]; Y = path2d[:,1]
                Z = np.full_like(X, z0)
                lines.append(np.column_stack([X,Y,Z]))
            k += 1
    return lines

def build_flowlines_straight(mesh, vhat, rake_xz, length, upstream, jitter):
    V = mesh.vertices; ctr = V.mean(axis=0)
    xmin,xmax = float(np.min(V[:,0])), float(np.max(V[:,0]))
    zmin,zmax = float(np.min(V[:,2])), float(np.max(V[:,2]))
    Dx, Dz = (xmax-xmin), (zmax-zmin)
    nx, nz = rake_xz
    xs = np.linspace(xmin-0.3*Dx, xmax+0.3*Dx, nx)
    zs = np.linspace(zmin-0.3*Dz, zmax+0.3*Dz, nz)
    jitx = (np.random.rand(nx*nz)-0.5)*jitter*Dx
    jitz = (np.random.rand(nx*nz)-0.5)*jitter*Dz
    D = max(Dx,Dz); base = ctr - upstream*D*vhat; L = length*D
    lines = []; k=0
    for xx in xs:
        for zz in zs:
            p0 = np.array([xx + jitx[k], base[1], zz + jitz[k]])
            p1 = p0 + L*vhat
            lines.append(np.vstack([p0, p1])); k+=1
    return lines

# === Visualization blocks ===

def visualize_main(mesh, vhat, Cp, method, show=False, save=False, save_prefix="aero",
                   lock_2d_to_global=True, invert_2d_y=False,
                   arrow2d="vertical", silhouette="both", sil_res=700,
                   cfd=None, projection_plane="XY"):
    if not (show or save): return
    if save and not show:
        try: plt.switch_backend("Agg")
        except Exception: pass

    V = mesh.vertices; F = mesh.faces; n = mesh.face_normals
    windward = (n @ (-vhat)) > 0.0
    Cp_norm = np.zeros_like(Cp, dtype=float)
    if np.any(windward):
        Cp_w = Cp[windward]; cmin = float(np.min(Cp_w)); cptp = float(np.ptp(Cp_w)) or 1.0
        Cp_norm[windward] = (Cp_w - cmin)/cptp

    # 3-D panel
    fig = plt.figure(figsize=(13,5.4))
    ax3 = fig.add_subplot(1,2,1, projection='3d')
    tris3d = [V[idx] for idx in F]
    face_colors = matplotlib.cm.inferno(np.clip(Cp_norm,0,1))
    face_colors[~windward,:] = [0.85,0.85,0.85,0.5]
    ax3.add_collection3d(Poly3DCollection(tris3d, facecolors=face_colors,
                                          linewidths=0.05, edgecolors=(0,0,0,0.05)))

    mins = V.min(axis=0); maxs = V.max(axis=0); ctr = (mins+maxs)/2.0; ext = (maxs-mins).max()
    for i,axis in enumerate(['x','y','z']):
        lo = ctr[i]-0.6*ext; hi = ctr[i]+0.6*ext
        getattr(ax3, f"set_{axis}lim")((lo,hi))
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.set_title(f"3D mesh — flow arrow (method: {method})")
    L = 0.4*ext
    ax3.quiver(ctr[0]-0.6*L*vhat[0], ctr[1]-0.6*L*vhat[1], ctr[2]-0.6*L*vhat[2],
               vhat[0], vhat[1], vhat[2], length=1.2*L, arrow_length_ratio=0.12,
               color='tab:cyan', linewidth=2)
    ax3.text(*(ctr + 0.7*L*vhat), "flow", color='tab:cyan')

    # 2-D panel (projection plane) with CFD overlay
    if lock_2d_to_global:
        pp = projection_plane.upper()
        if pp == "XY":
            P = V[:, [0,1]]; xlab, ylab = 'X (global)', 'Y (global)'
        elif pp == "XZ":
            P = V[:, [0,2]]; xlab, ylab = 'X (global)', 'Z (global)'
        elif pp == "YZ":
            P = V[:, [1,2]]; xlab, ylab = 'Y (global)', 'Z (global)'
        else:
            P = V[:, [0,1]]; xlab, ylab = 'X (global)', 'Y (global)'
    else:
        u_b, w_b, _ = ortho_basis_from(vhat)
        P = np.column_stack([V @ u_b, V @ w_b]); xlab, ylab = 'u (⊥ flow)', 'w (⊥ flow)'

    ax2 = fig.add_subplot(1,2,2)
    polys2d = [P[idx] for idx in F]
    face_colors2d = matplotlib.cm.inferno(np.clip(Cp_norm,0,1))
    face_colors2d[~windward,:] = [0.9,0.9,0.9,0.4]
    ax2.add_collection(PolyCollection(polys2d, facecolors=face_colors2d,
                                      linewidths=0.05, edgecolors=(0,0,0,0.05)))
    mins2 = P.min(axis=0); maxs2 = P.max(axis=0); ctr2 = (mins2+maxs2)/2.0; ext2 = (maxs2-mins2).max()
    ax2.set_xlim(ctr2[0]-0.6*ext2, ctr2[0]+0.6*ext2)
    ax2.set_ylim(ctr2[1]-0.6*ext2, ctr2[1]+0.6*ext2)
    if invert_2d_y: ax2.invert_yaxis()
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel(xlab); ax2.set_ylabel(ylab)
    ax2.set_title("Projection (Cp windward shaded)")

    legend = []
    # overlay CFD Mach + streamlines, with rotation so +Y is UP
    if cfd is not None:
        xs, ys, solid, U = cfd
        rho,u,v,p,a = cons2prim(U, GAMMA)
        M = np.sqrt(np.maximum(u*u+v*v,0.0))/np.maximum(a,1e-12)
        xs_v, ys_v, Mv, solid_v = orient_xy_for_plot(xs, ys, M, solid, CFD_ROTATE_VIEW)
        Xv, Yv = np.meshgrid(xs_v, ys_v, indexing='xy')
        pcm = ax2.pcolormesh(Xv, Yv, np.ma.masked_where(solid_v, Mv),
                             shading='nearest', cmap='viridis')
        cb = fig.colorbar(pcm, ax=ax2, pad=0.01); cb.set_label('Mach (CFD)')

        if STREAMLINES_ON_FIELDS:
            xs_u, ys_u, u_plot, v_plot, solid_vec = orient_vector_field(
                xs, ys, u, v, solid, CFD_ROTATE_VIEW
            )
            ax2.streamplot(xs_u, ys_u,
                           np.where(solid_vec, np.nan, u_plot),
                           np.where(solid_vec, np.nan, v_plot),
                           density=STREAMLINE_DENSITY, linewidth=0.8,
                           arrowsize=1.0, minlength=0.2)

        ax2.contour(Xv, Yv, solid_v.astype(float), levels=[0.5], colors=['lime'], linewidths=1.5)
        legend.append(Line2D([0],[0], color='lime', lw=2, label='silhouette (CFD)'))

    # flow arrow on 2-D panel
    if ARROW2D.lower().startswith('vert'):
        ax2.annotate('', xy=(ctr2[0], ctr2[1]+0.45*ext2), xytext=(ctr2[0], ctr2[1]-0.45*ext2),
                     arrowprops=dict(arrowstyle='->', lw=2, color='tab:cyan'))
        ax2.text(ctr2[0], ctr2[1]+0.47*ext2, "flow ↑", color='tab:cyan', ha='center', va='bottom')
    else:
        ax2.annotate('', xy=(ctr2[0]+0.45*ext2, ctr2[1]), xytext=(ctr2[0]-0.45*ext2, ctr2[1]),
                     arrowprops=dict(arrowstyle='->', lw=2, color='tab:cyan'))
        ax2.text(ctr2[0]+0.47*ext2, ctr2[1], "flow →", color='tab:cyan', va='center')

    if legend: ax2.legend(handles=legend, loc='upper right', framealpha=0.85)

    plt.tight_layout()
    if save:
        fn = f"{save_prefix}_overview.png"
        plt.savefig(fn, dpi=170); print(f"[vis] Saved {fn}")
    if show: plt.show()
    plt.close(fig)

def render_3d_views(mesh, vhat, Cp, views, save=False, save_prefix="aero"):
    if not (save or SHOW_PLOTS): return
    V = mesh.vertices; F = mesh.faces; n=mesh.face_normals
    windward = (n @ (-vhat)) > 0.0
    Cp_norm = np.zeros_like(Cp, dtype=float)
    if np.any(windward):
        Cp_w = Cp[windward]; cmin = float(np.min(Cp_w)); cptp = float(np.ptp(Cp_w)) or 1.0
        Cp_norm[windward] = (Cp_w - cmin)/cptp
    tris3d = [V[idx] for idx in F]
    mins = V.min(axis=0); maxs = V.max(axis=0); ctr = (mins+maxs)/2.0; ext = (maxs-mins).max()
    L = 0.4*ext
    for name, elev, azim in views:
        fig = plt.figure(figsize=(6.5,6.2))
        ax = fig.add_subplot(1,1,1, projection='3d')
        face_colors = matplotlib.cm.inferno(np.clip(Cp_norm,0,1))
        face_colors[~windward,:] = [0.85,0.85,0.85,0.5]
        ax.add_collection3d(Poly3DCollection(tris3d, facecolors=face_colors,
                                             linewidths=0.05, edgecolors=(0,0,0,0.05)))
        for i,axis in enumerate(['x','y','z']):
            lo = ctr[i]-0.6*ext; hi = ctr[i]+0.6*ext
            getattr(ax, f"set_{axis}lim")((lo,hi))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name}  (elev={elev}, azim={azim})")
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.quiver(ctr[0]-0.6*L*vhat[0], ctr[1]-0.6*L*vhat[1], ctr[2]-0.6*L*vhat[2],
                  vhat[0], vhat[1], vhat[2], length=1.2*L, arrow_length_ratio=0.12,
                  color='tab:cyan', linewidth=2)
        fn = f"{save_prefix}_3d_{name}.png"
        if save: plt.savefig(fn, dpi=170); print(f"[vis] Saved {fn}")
        if SHOW_PLOTS: plt.show()
        plt.close(fig)

def render_3d_thermal(mesh, vhat, q_stag, views, save=False, save_prefix="aero"):
    if not (THERMAL and (save or SHOW_PLOTS)): return
    V = mesh.vertices; F = mesh.faces
    qf = thermal_face_proxy(mesh, vhat, q_stag)
    tris3d = [V[idx] for idx in F]
    mins = V.min(axis=0); maxs = V.max(axis=0); ctr = (mins+maxs)/2.0; ext = (maxs-mins).max()
    for name, elev, azim in views:
        fig = plt.figure(figsize=(6.5,6.2))
        ax = fig.add_subplot(1,1,1, projection='3d')
        face_colors = matplotlib.cm.plasma(np.clip(qf/np.nanmax(qf + 1e-16), 0, 1))
        ax.add_collection3d(Poly3DCollection(tris3d, facecolors=face_colors,
                                             linewidths=0.05, edgecolors=(0,0,0,0.05)))
        for i,axis in enumerate(['x','y','z']):
            lo = ctr[i]-0.6*ext; hi = ctr[i]+0.6*ext
            getattr(ax, f"set_{axis}lim")((lo,hi))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name} (thermal proxy)")
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        m = matplotlib.cm.ScalarMappable(cmap='plasma'); m.set_array(qf)
        cb = plt.colorbar(m, ax=ax, fraction=0.046, pad=0.04); cb.set_label('thermal proxy (rel.)')
        fn = f"{save_prefix}_3d_{name}.png"
        if save: plt.savefig(fn, dpi=170); print(f"[vis] Saved {fn}")
        if SHOW_PLOTS: plt.show()
        plt.close(fig)

def render_3d_flowlines(mesh, vhat, views, save=False, save_prefix="aero",
                        rake_xz=(12,8), length=2.2, upstream=1.0, jitter=0.1,
                        mode="straight", cfd=None, dt_frac=0.35, steps=900):
    if not (save or SHOW_PLOTS): return
    V = mesh.vertices; F = mesh.faces
    if mode == "cfd" and cfd is not None:
        xs, ys, solid, U = cfd
        rho,u,v,p,a = cons2prim(U, GAMMA)
        lines = build_flowlines_from_cfd_xy(mesh, xs, ys, u, v, solid,
                                            rake_xz=rake_xz, dt_frac=dt_frac,
                                            steps=steps, upstream=upstream,
                                            jitter=jitter)
    else:
        lines = build_flowlines_straight(mesh, vhat, rake_xz, length, upstream, jitter)

    tris3d = [V[idx] for idx in F]
    mins = V.min(axis=0); maxs = V.max(axis=0); ctr = (mins+maxs)/2.0; ext = (maxs-mins).max()
    for name, elev, azim in views:
        fig = plt.figure(figsize=(6.5,6.2))
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.add_collection3d(Poly3DCollection(tris3d, facecolors=(0.9,0.9,0.9,0.35),
                                             linewidths=0.05, edgecolors=(0,0,0,0.05)))
        for L in lines:
            ax.plot(L[:,0], L[:,1], L[:,2], lw=1.0, alpha=0.9, color='tab:blue')
        for i,axis in enumerate(['x','y','z']):
            lo = ctr[i]-0.6*ext; hi = ctr[i]+0.6*ext
            getattr(ax, f"set_{axis}lim")((lo,hi))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name} (flow lines: {mode})")
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        fn = f"{save_prefix}_{name}.png"
        if save: plt.savefig(fn, dpi=170); print(f"[vis] Saved {fn}")
        if SHOW_PLOTS: plt.show()
        plt.close(fig)

def extra_cfd_field_plots(xs, ys, solid, U, keys, save=False, save_prefix="aero",
                          stream=True, rotate_deg=0):
    rho,u,v,p,a = cons2prim(U, GAMMA)
    speed = np.sqrt(np.maximum(u*u+v*v,0.0))
    M = speed / np.maximum(a,1e-12)
    drdx = (np.roll(rho,-1,axis=1) - np.roll(rho,1,axis=1)) / (2*(xs[1]-xs[0]))
    drdy = (np.roll(rho,-1,axis=0) - np.roll(rho,1,axis=0)) / (2*(ys[1]-ys[0]))
    gr = np.sqrt(drdx*drdx + drdy*drdy)
    field_bank = {
        "mach":     (M,      "Mach"),
        "pressure": (p,      "Pressure (nd)"),
        "density":  (rho,    "Density (nd)"),
        "speed":    (speed,  "|V| (nd)"),
        "gradrho":  (gr,     "|∇ρ| (nd)"),
    }
    for key in keys:
        if key not in field_bank: continue
        A, title = field_bank[key]
        xs_v, ys_v, Av, solid_v = orient_xy_for_plot(xs, ys, A, solid, rotate_deg)
        Xv, Yv = np.meshgrid(xs_v, ys_v, indexing='xy')
        fig, ax = plt.subplots(figsize=(7.8,4.8))
        pcm = ax.pcolormesh(Xv, Yv, np.ma.masked_where(solid_v, Av),
                            shading='nearest', cmap='viridis')
        if stream:
            xs_u, ys_u, u_plot, v_plot, solid_vec = orient_vector_field(
                xs, ys, u, v, solid, rotate_deg
            )
            ax.streamplot(xs_u, ys_u,
                          np.where(solid_vec,np.nan,u_plot),
                          np.where(solid_vec,np.nan,v_plot),
                          density=STREAMLINE_DENSITY, linewidth=0.7,
                          arrowsize=0.9, minlength=0.2)

        ax.contour(Xv, Yv, solid_v.astype(float), levels=[0.5], colors=['white'], linewidths=1.0)

        # For Mach plot, highlight sonic line
        if key == "mach":
            try:
                ax.contour(Xv, Yv, Av, levels=[1.0], colors='k', linewidths=1.0)
            except Exception:
                pass

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_title(f"{title}")
        cb = plt.colorbar(pcm, ax=ax, pad=0.01); cb.set_label(title)
        if save:
            fn = f"{save_prefix}_cfd_{key}_rot{int(rotate_deg)}.png"
            plt.savefig(fn, dpi=170); print(f"[vis] Saved {fn}")
        if SHOW_PLOTS: plt.show()
        plt.close(fig)

def pressure_views_multi(xs, ys, solid, U, rotations, save=False, save_prefix="aero",
                         stream=True):
    for ang in rotations:
        extra_cfd_field_plots(xs, ys, solid, U, keys=["pressure"],
                              save=save, save_prefix=save_prefix,
                              stream=stream, rotate_deg=ang)

def export_colored_ply(mesh, Cp, vhat, outfile="flow_cp.ply"):
    n = mesh.face_normals; windward = (n @ (-vhat)) > 0.0
    Cp_norm = np.zeros_like(Cp, dtype=float)
    if np.any(windward):
        Cp_w = Cp[windward]; cmin = float(np.min(Cp_w)); cptp = float(np.ptp(Cp_w)) or 1.0
        Cp_norm[windward] = (Cp_w - cmin)/cptp
    rgba = (matplotlib.cm.inferno(np.clip(Cp_norm,0,1))*255).astype(np.uint8)
    rgba[~windward,:] = np.array([220,220,220,180], dtype=np.uint8)
    m = mesh.copy(); m.visual.face_colors = rgba; m.export(outfile)
    print(f"[ply] Wrote {outfile}")

# === AoA sweep helpers ===

def rotate_flow_about_x(vhat, alpha_deg):
    # Nose-up alpha rotates the wind vector about +X (right-hand rule).
    a = math.radians(alpha_deg)
    ca, sa = math.cos(a), math.sin(a)
    Rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
    return unit(Rx @ vhat)

def face_forces_pressure(mesh, vhat, Cp, Aref):
    # Dimensional per-unit-q∞ force on each face (N/q∞), and moment about origin.
    n = mesh.face_normals
    A = mesh.area_faces
    F_faces = - (Cp[:,None] * n) * A[:,None]
    r_fc    = mesh.triangles_center
    M_faces = np.cross(r_fc, F_faces)
    F = F_faces.sum(axis=0)
    M = M_faces.sum(axis=0)
    Cvec = F / Aref
    return Cvec, F, M

def cm_about_point(mesh, vhat, Cp, Aref, ref_pt, L_ref):
    Cvec, Fnd, Mnd = face_forces_pressure(mesh, vhat, Cp, Aref)
    r0 = ref_pt
    M_nd_about_r0 = Mnd - np.cross(r0, Fnd)
    C_mx = M_nd_about_r0[0] / (Aref * L_ref)
    return C_mx

# =============================================================================
# ================================== MAIN =====================================
# =============================================================================
def main():
    # Load & scale
    mesh = trimesh.load(STL_PATH)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    if UNITS in UNIT_SCALE: mesh.apply_scale(UNIT_SCALE[UNITS])
    if SCALE != 1.0: mesh.apply_scale(SCALE)

    # Orient: map CAD_UP → +Z, then yaw spin, then recenter
    cad_up_vec = parse_axis(CAD_UP)
    R = rotation_matrix_from_vectors(cad_up_vec, np.array([0.0,0.0,1.0]))
    T = np.eye(4); T[:3,:3] = R; mesh.apply_transform(T)
    if abs(CAD_YAW_DEG) > 1e-12: mesh.apply_transform(euler_matrix_z(CAD_YAW_DEG))
    recenter(mesh, CENTER)

    # Flow vector v̂
    if FLOW_MODE.upper() == "ALONG_UP":
        vhat = np.array([0.0,0.0, 1.0 if FLOW_SENSE.lower()=="up" else -1.0])
    else:
        vhat = rot_xyz(parse_flow(FLOW), yaw=YAW, pitch=PITCH, roll=ROLL); vhat = unit(vhat)

    # Bounding box report
    bb_min, bb_max = mesh.bounds; ext = bb_max - bb_min
    print(f"[mesh] bbox min (m): {bb_min}")
    print(f"[mesh] bbox max (m): {bb_max}")
    print(f"[mesh] extents (m):  {ext}")
    print(f"[orient] CAD_UP '{CAD_UP}' → +Z, yaw {CAD_YAW_DEG}°")
    print(f"[flow] v̂ = [{vhat[0]:+.4f}, {vhat[1]:+.4f}, {vhat[2]:+.4f}] (drag along +v̂)")

    # Newtonian aero
    Cd_p, Cvec_p, Aref, r_cp, Cp = newtonian_coeffs(mesh, vhat, method=METHOD, gamma=GAMMA, M=MACH)
    rho = P_INF/(R_AIR*T_INF); a = math.sqrt(GAMMA*R_AIR*T_INF); V = MACH*a; q = 0.5*rho*V*V
    S_wet = float(mesh.area); L_flow = np.ptp(mesh.vertices @ vhat)
    if INCLUDE_FRICTION: Cd_f, Cf = friction_cd(S_wet, Aref, L_flow, rho, V, MACH, model=CF_MODEL)
    else: Cd_f, Cf = 0.0, 0.0
    Cd_tot = Cd_p + Cd_f
    u_b, w_b, _ = ortho_basis_from(vhat)
    Cl = float(Cvec_p @ u_b)
    Cs = float(Cvec_p @ w_b)
    LD = (abs(Cl)/Cd_tot) if Cd_tot>1e-12 else float('nan')
    D_p = Cd_p*q*Aref; D_f = Cd_f*q*Aref; D_tot = D_p + D_f
    VOL, vhow = safe_volume(mesh)
    Deq = 2.0*math.sqrt(Aref/math.pi) if Aref>0 else float('nan')

    # Print summary
    print("\n[Aero from STL: {}]".format(METHOD))
    print(f"  STL: {STL_PATH}  |  units: {UNITS}  |  scale: {SCALE}")
    print(f"  Centering: {CENTER}")
    print(f"  A_ref (projected): {Aref:.6g} m^2  |  D_eq: {Deq:.6g} m")
    print(f"  Wetted area:       {S_wet:.6g} m^2  |  Volume: {VOL:.6g} m^3 (method={vhow})")
    print(f"\n  C_D (pressure):    {Cd_p:.6f}")
    if INCLUDE_FRICTION:
        print(f"  C_D (friction):    {Cd_f:.6f}   [Cf={Cf:.5f}, L_flow={L_flow:.3g} m]")
    print(f"  C_D (total):       {Cd_tot:.6f}")
    print(f"  C_vec (Cx,Cy,Cz):  [{Cvec_p[0]:+.6f}, {Cvec_p[1]:+.6f}, {Cvec_p[2]:+.6f}]")
    print(f"  r_CP (m):          [{r_cp[0]:.6g}, {r_cp[1]:.6g}, {r_cp[2]:.6g}]")
    print(f"  Cl, Cs, L/D:       {Cl:.6f}, {Cs:.6f}, {LD:.3f}")

    print("\n[Freestream]")
    print(f"  γ={GAMMA}, M={MACH}, T={T_INF} K, p={P_INF} Pa  →  a={a:.3f} m/s, "
          f"V={V:.3f} m/s, ρ={rho:.6g} kg/m³, q∞={q:.6g} Pa")
    print(f"  Drag D_p, D_f, D:  {D_p:.6g} N, {D_f:.6g} N, {D_tot:.6g} N")

    # Thermal proxies
    if THERMAL:
        Rn = NOSE_RADIUS if NOSE_RADIUS>0 else estimate_nose_radius(mesh, vhat)
        q_stag = sutton_graves_qdot_Wm2(rho, V, Rn)
        rrec = (PRANDTL**(1/3)) if RECOVERY=="turbulent" else math.sqrt(PRANDTL)
        Taw  = T_INF*(1.0 + rrec*0.5*(GAMMA-1.0)*MACH*MACH)
        print("\n[Thermal proxies]")
        print(f"  Nose radius R_n:   {Rn:.6g} m  ({'override' if NOSE_RADIUS>0 else 'estimated'})")
        print(f"  q̇_stag (Sutton–Graves): {q_stag:.6g} W/m²")
        print(f"  Taw (adiabatic):   {Taw:.3f} K   [recovery={RECOVERY}, Pr={PRANDTL}]")
    else:
        q_stag = float('nan')

    # CFD on chosen projection plane
    cfd_tuple = None
    cfd_diag  = None
    if ENABLE_CFD:
        pp = PROJECTION_PLANE.upper()
        if pp == "XY":
            P2 = mesh.vertices[:, [0,1]]
        elif pp == "XZ":
            P2 = mesh.vertices[:, [0,2]]
        elif pp == "YZ":
            P2 = mesh.vertices[:, [1,2]]
        else:
            P2 = mesh.vertices[:, [0,1]]

        hull = convex_hull_2d(P2)
        bounds = np.vstack([P2.min(axis=0), P2.max(axis=0)])
        print("\n[CFD] starting 2-D Euler HLLE … (+Y inflow; display rotated so flow is bottom→top)")
        print(f"      using projection plane {PROJECTION_PLANE} for silhouette")
        xs, ys, solid, U = run_cfd_on_silhouette(
            hull, bounds, GAMMA, MACH,
            nx=CFD_NX, ny=CFD_NY,
            box_expand_lr=CFD_BOX_EXPAND[0], box_expand_ud=CFD_BOX_EXPAND[1],
            steps=STEPS, cfl=CFL, tol=RES_TOL, report_every=REPORT_EVERY
        )
        cfd_tuple = (xs, ys, solid, U)
        cfd_diag  = cfd_summary_report(xs, ys, solid, U, Deq, GAMMA)

    # Renders
    if PLOT_OVERVIEW:
        visualize_main(mesh, vhat, Cp, METHOD,
                       show=SHOW_PLOTS, save=SAVE_PLOTS, save_prefix=SAVE_PREFIX,
                       lock_2d_to_global=LOCK_2D_TO_GLOBAL, invert_2d_y=INVERT_2D_Y,
                       arrow2d=ARROW2D, silhouette=SILHOUETTE, sil_res=SIL_RES,
                       cfd=cfd_tuple, projection_plane=PROJECTION_PLANE)

    if PLOT_3D_ANGLES:
        render_3d_views(mesh, vhat, Cp, VIEWS_3D,
                    save=SAVE_PLOTS,
                    save_prefix=SAVE_PREFIX)

    if PLOT_3D_THERMAL and THERMAL:
        render_3d_thermal(mesh, vhat, q_stag, THERMAL_VIEWS_3D,
                          save=SAVE_PLOTS, save_prefix=SAVE_PREFIX)

    if PLOT_3D_FLOWLINES:
        render_3d_flowlines(
            mesh, vhat, FLOWLINES_VIEWS_3D,
            save=SAVE_PLOTS, save_prefix=SAVE_PREFIX,
            rake_xz=FLOWLINES_RAKE_XZ, length=FLOWLINES_LEN,
            upstream=FLOWLINES_OFFSET, jitter=FLOWLINES_JITTER,
            mode=FLOWLINES_MODE, cfd=cfd_tuple,
            dt_frac=FLOWLINES_DT, steps=FLOWLINES_STEPS
        )

    if ENABLE_CFD and PLOT_CFD_FIELDS and cfd_tuple is not None:
        xs, ys, solid, U = cfd_tuple
        extra_cfd_field_plots(xs, ys, solid, U, CFD_FIELD_KEYS,
                              save=SAVE_PLOTS, save_prefix=SAVE_PREFIX,
                              stream=STREAMLINES_ON_FIELDS, rotate_deg=CFD_ROTATE_VIEW)

    if ENABLE_CFD and PLOT_CFD_PRESSURE_VIEWS and cfd_tuple is not None:
        xs, ys, solid, U = cfd_tuple
        pressure_views_multi(xs, ys, solid, U, CFD_PRESSURE_ROTATIONS,
                             save=SAVE_PLOTS, save_prefix=SAVE_PREFIX,
                             stream=True)

    if ENABLE_CFD and PLOT_CENTERLINE and cfd_tuple is not None:
        xs, ys, solid, U = cfd_tuple
        rho,u,v,p,a = cons2prim(U, GAMMA)
        M = np.sqrt(np.maximum(u*u+v*v,0.0))/np.maximum(a,1e-12)

        dx = xs[1] - xs[0]

        if cfd_diag is not None and "ix" in cfd_diag:
            ix = int(cfd_diag["ix"])
        else:
            xmid = 0.5*(xs[0] + xs[-1])
            ix = int(np.clip(round((xmid - xs[0]) / dx), 0, len(xs)-1))

        fig, ax = plt.subplots(figsize=(6.6,4.2))
        ax.plot(ys, M[:,ix], label='Mach')

        ax2 = ax.twinx()
        ax2.plot(ys, p[:,ix], color='tab:orange', label='p (nd)')

        if cfd_diag is not None:
            y_shock = cfd_diag.get("y_shock", float('nan'))
            y_nose  = cfd_diag.get("y_nose",  float('nan'))
            labels = []
            if np.isfinite(y_shock):
                ax.axvline(y_shock, color='k', linestyle='--', alpha=0.8)
                labels.append("shock (|∇ρ| max)")
            if np.isfinite(y_nose):
                ax.axvline(y_nose,  color='r', linestyle=':',  alpha=0.8)
                labels.append("nose")

            if labels:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2,
                          labels1 + labels2 + labels,
                          loc='best', framealpha=0.85)

        ax.set_xlabel('y (solver coords)')
        ax.set_title('Centerline (x ≈ body mid) Mach & p')
        ax.set_ylabel('Mach')
        ax2.set_ylabel('p (nd)')

        if SAVE_PLOTS:
            fn = f"{SAVE_PREFIX}_centerline.png"
            plt.savefig(fn, dpi=170); print(f"[vis] Saved {fn}")
        if SHOW_PLOTS: plt.show()
        plt.close(fig)

    if EXPORT_PLY:
        export_colored_ply(mesh, Cp, vhat, outfile=EXPORT_PLY)

    # ---------------- AoA sweep ----------------------------------------------
    if DO_ALPHA_SWEEP:
        rows = []
        print("\n[α-sweep] α (deg) |  Cd    |  Cl(u) |  Cm_x  |  r_cp_x   r_cp_y   r_cp_z")
        for a_deg in ALPHAS_DEG:
            vhat_a = rotate_flow_about_x(vhat, a_deg)
            Cd_p_a, Cvec_p_a, Aref_a, r_cp_a, Cp_a = newtonian_coeffs(
                mesh, vhat_a, method=METHOD, gamma=GAMMA, M=MACH
            )
            u_ba, w_ba, _ = ortho_basis_from(vhat_a)
            Cl_a = float(Cvec_p_a @ u_ba)
            Cm_x = cm_about_point(mesh, vhat_a, Cp_a, Aref_a,
                                  ref_pt=np.zeros(3), L_ref=L_REF_MOMENT)
            rows.append([a_deg, Cd_p_a, Cl_a, Cm_x,
                         r_cp_a[0], r_cp_a[1], r_cp_a[2]])
            print(f"  {a_deg:>6.1f}    {Cd_p_a:6.4f}  {Cl_a:6.4f}  {Cm_x:7.4f}   "
                  f"{r_cp_a[0]:.4g} {r_cp_a[1]:.4g} {r_cp_a[2]:.4g}")

        if SAVE_SWEEP_CSV:
            with open(SWEEP_CSV_NAME, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["alpha_deg","Cd","Cl_u","Cm_x","r_cp_x","r_cp_y","r_cp_z"])
                w.writerows(rows)
            print(f"[sweep] wrote {SWEEP_CSV_NAME}")

        if PLOT_SWEEP and (SHOW_PLOTS or SAVE_PLOTS):
            al = [r[0] for r in rows]
            cd = [r[1] for r in rows]
            cl = [r[2] for r in rows]
            cm = [r[3] for r in rows]
            fig, ax = plt.subplots(1,1, figsize=(7.0,4.6))
            ax.plot(al, cd, 'o-', label='Cd')
            ax.plot(al, cl, 'o-', label='Cl (u-axis)')
            ax.plot(al, cm, 'o-', label='Cm_x')
            ax.set_xlabel('alpha (deg)'); ax.grid(True, alpha=0.3)
            ax.legend(); ax.set_title('α sweep (Newtonian)')
            if SAVE_PLOTS:
                fn = f"{SAVE_PREFIX}_alpha_sweep.png"
                plt.savefig(fn, dpi=170); print(f"[vis] Saved {fn}")
            if SHOW_PLOTS: plt.show()
            plt.close(fig)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e); sys.exit(1)
