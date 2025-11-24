#!/usr/bin/env python
"""
PMP GUI – Piper Mission Planner
===============================

Portable Tkinter-based GUI frontend for the PMP framework.

Features (initial):
- Mission tab:
    * Central body, origin & target bodies
    * Departure & arrival UTC
    * Simple mission name / notes
- Transfer tab:
    * Lambert transfer config
    * Run transfer and show Δv, TOF, v_inf, MOI Δv
- Probe Env / Power tab:
    * Probe environment, thermal, battery, panel config
    * Run surface simulation and show summary
- Orbiter EPS tab:
    * Orbit + lighting config
    * Array, battery, loads
    * Run EPS sim and show per-orbit energy balance
- EDL tab:
    * Body atmosphere, aeroshell, parachute, retro
    * Run 1D EDL and show touchdown velocity / events
- Comms tab:
    * Probe/orbiter link configs
    * Deep-space X-band config
    * Run quick link budget
- ADCS tab:
    * Inertia, RW cluster, PD gains
    * Run simple slew and show energy / settling time
- Summary tab:
    * Collects last-run results summaries in one place

NOTE:
- This is a *framework* GUI: lots of knobs, many fields, and hooks
  into PMP functions. You can expand or rearrange subsystems easily.

"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import datetime as dt

# Optional numeric stuff; GUI will still launch if not installed, but
# some actions will fail gracefully.
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

# Try to import PMP modules. All these are optional; GUI will check.
PMP_IMPORT_ERRORS = {}

def _try_import():
    global PMP_IMPORT_ERRORS
    try:
        from pmp.trajectory.transfer import EndpointConfig, TransferConfig, compute_lambert_transfer
    except Exception as e:  # pragma: no cover
        PMP_IMPORT_ERRORS["transfer"] = str(e)
    try:
        from pmp.seasons.mars_seasons import get_arrival_ls_and_season
    except Exception as e:  # pragma: no cover
        PMP_IMPORT_ERRORS["seasons"] = str(e)
    try:
        from pmp.power.probe_env import (
            ProbeEnvConfig,
            ProbePanelConfig,
            ProbeBatteryConfig,
            ProbeThermalConfig,
            ProbeEnvSimConfig,
            simulate_probe_env_power,
        )
    except Exception as e:  # pragma: no cover
        PMP_IMPORT_ERRORS["probe_env"] = str(e)
    try:
        from pmp.power.orbiter_eps import (
            OrbitLightingConfig,
            ArrayConfig,
            BatteryConfig,
            LoadConfig,
            OrbiterEPSSimConfig,
            simulate_orbiter_eps,
            magpie_default_eps_config,
        )
    except Exception as e:  # pragma: no cover
        PMP_IMPORT_ERRORS["eps"] = str(e)
    try:
        from pmp.edl.edl_1d import (
            BodyAtmosphere,
            VehicleAeroshell,
            ParachuteConfig,
            RetroConfig,
            EDLSimConfig,
            simulate_edl,
        )
    except Exception as e:  # pragma: no cover
        PMP_IMPORT_ERRORS["edl"] = str(e)
    try:
        from pmp.comms.link_mars import (
            SurfaceProbeRadio,
            OrbiterRadio,
            StaticPassConfig,
            compute_static_probe_orbiter_link,
            magpie_default_probe_433,
            magpie_default_probe_915,
            magpie_default_orbiter_radio,
            DeepSpaceLinkConfig,
            compute_deep_space_link,
            magpie_default_deep_space_cfg,
        )
    except Exception as e:  # pragma: no cover
        PMP_IMPORT_ERRORS["comms"] = str(e)
    try:
        from pmp.trajectory.adcs import (
            RWClusterConfig,
            RigidBodyConfig,
            SlewPDConfig,
            ADCSSlewSimConfig,
            simulate_pd_slew,
            magpie_default_orbiter_body,
            magpie_default_orbiter_rw_cluster,
        )
    except Exception as e:  # pragma: no cover
        PMP_IMPORT_ERRORS["adcs"] = str(e)


_try_import()


class PMPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Piper Mission Planner (PMP) – GUI")
        self.geometry("1150x800")
        self.minsize(1000, 700)

        # store last results summaries for the Summary tab
        self.summary_entries = []

        self._build_menu()
        self._build_notebook()

    # ------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Mission Config...", command=self.save_config)
        file_menu.add_command(label="Load Mission Config...", command=self.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Show PMP Import Status", command=self.show_import_status)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def show_import_status(self):
        if not PMP_IMPORT_ERRORS:
            messagebox.showinfo("PMP Imports", "All PMP subsystems imported successfully.")
            return
        msg_lines = ["Some PMP modules could not be imported:\n"]
        for k, v in PMP_IMPORT_ERRORS.items():
            msg_lines.append(f"- {k}: {v}")
        messagebox.showwarning("PMP Imports", "\n".join(msg_lines))

    def show_about(self):
        messagebox.showinfo(
            "About PMP",
            "Piper Mission Planner (PMP)\n"
            "Framework + GUI for space mission design.\n\n"
            "This GUI is a portable Tkinter frontend.\n"
            "Use the tabs to explore transfer, power, EDL,\n"
            "comms, ADCS, and more.\n"
        )

    # ------------------------------------------------------------
    # Notebook + tabs
    # ------------------------------------------------------------
    def _build_notebook(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.mission_tab = MissionTab(nb, self)
        self.transfer_tab = TransferTab(nb, self)
        self.probe_tab = ProbeEnvTab(nb, self)
        self.eps_tab = OrbiterEPSTab(nb, self)
        self.edl_tab = EDLTab(nb, self)
        self.comms_tab = CommsTab(nb, self)
        self.adcs_tab = ADCSTab(nb, self)
        self.summary_tab = SummaryTab(nb, self)

        nb.add(self.mission_tab, text="Mission")
        nb.add(self.transfer_tab, text="Transfer")
        nb.add(self.probe_tab, text="Probe Env/Power")
        nb.add(self.eps_tab, text="Orbiter EPS")
        nb.add(self.edl_tab, text="EDL")
        nb.add(self.comms_tab, text="Comms")
        nb.add(self.adcs_tab, text="ADCS")
        nb.add(self.summary_tab, text="Summary")

        self.notebook = nb

    # ------------------------------------------------------------
    # Config save/load (JSON)
    # ------------------------------------------------------------
    def collect_config(self) -> dict:
        return {
            "mission": self.mission_tab.get_state(),
            "transfer": self.transfer_tab.get_state(),
            "probe_env": self.probe_tab.get_state(),
            "eps": self.eps_tab.get_state(),
            "edl": self.edl_tab.get_state(),
            "comms": self.comms_tab.get_state(),
            "adcs": self.adcs_tab.get_state(),
        }

    def apply_config(self, cfg: dict):
        self.mission_tab.set_state(cfg.get("mission", {}))
        self.transfer_tab.set_state(cfg.get("transfer", {}))
        self.probe_tab.set_state(cfg.get("probe_env", {}))
        self.eps_tab.set_state(cfg.get("eps", {}))
        self.edl_tab.set_state(cfg.get("edl", {}))
        self.comms_tab.set_state(cfg.get("comms", {}))
        self.adcs_tab.set_state(cfg.get("adcs", {}))

    def save_config(self):
        cfg = self.collect_config()
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Mission Config",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        messagebox.showinfo("Save Config", f"Config saved to:\n{path}")

    def load_config(self):
        path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Mission Config",
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.apply_config(cfg)
        messagebox.showinfo("Load Config", f"Config loaded from:\n{path}")

    # ------------------------------------------------------------
    # Summary tab helper
    # ------------------------------------------------------------
    def add_summary_entry(self, title: str, data: dict):
        self.summary_entries.append((title, data))
        self.summary_tab.refresh()


# ------------------------------------------------------------
# Mission tab
# ------------------------------------------------------------
class MissionTab(ttk.Frame):
    def __init__(self, parent, app: PMPApp):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(outer)
        right = ttk.Frame(outer)
        left.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="both", expand=True)

        # Mission basic info
        g1 = ttk.LabelFrame(left, text="Mission Info")
        g1.pack(fill="x", padx=5, pady=5)

        self.mission_name = tk.StringVar(value="MAGPIE Baseline")
        self.mission_notes = tk.StringVar(value="Demo mission config.")

        ttk.Label(g1, text="Mission Name:").grid(row=0, column=0, sticky="w")
        ttk.Entry(g1, textvariable=self.mission_name, width=32).grid(row=0, column=1, sticky="ew")

        ttk.Label(g1, text="Notes:").grid(row=1, column=0, sticky="nw")
        ttk.Entry(g1, textvariable=self.mission_notes, width=50).grid(row=1, column=1, sticky="ew")

        g1.columnconfigure(1, weight=1)

        # Bodies and dates
        g2 = ttk.LabelFrame(left, text="Bodies & Dates")
        g2.pack(fill="x", padx=5, pady=5)

        self.central_body = tk.StringVar(value="Sun")
        self.origin_body = tk.StringVar(value="Earth")
        self.target_body = tk.StringVar(value="Mars")

        self.departure_utc = tk.StringVar(value="2028-11-10 00:00:00")
        self.arrival_utc = tk.StringVar(value="2029-08-23 00:00:00")

        ttk.Label(g2, text="Central Body:").grid(row=0, column=0, sticky="w")
        ttk.Entry(g2, textvariable=self.central_body, width=15).grid(row=0, column=1, sticky="w")

        ttk.Label(g2, text="Origin Body:").grid(row=1, column=0, sticky="w")
        ttk.Entry(g2, textvariable=self.origin_body, width=15).grid(row=1, column=1, sticky="w")

        ttk.Label(g2, text="Target Body:").grid(row=2, column=0, sticky="w")
        ttk.Entry(g2, textvariable=self.target_body, width=15).grid(row=2, column=1, sticky="w")

        ttk.Label(g2, text="Departure UTC:").grid(row=0, column=2, sticky="w")
        ttk.Entry(g2, textvariable=self.departure_utc, width=20).grid(row=0, column=3, sticky="w")

        ttk.Label(g2, text="Arrival UTC:").grid(row=1, column=2, sticky="w")
        ttk.Entry(g2, textvariable=self.arrival_utc, width=20).grid(row=1, column=3, sticky="w")

        for c in range(4):
            g2.columnconfigure(c, weight=1)

        # Right side: mission summary / notes
        g3 = ttk.LabelFrame(right, text="Description")
        g3.pack(fill="both", expand=True, padx=5, pady=5)

        self.mission_desc = tk.Text(g3, height=20, wrap="word")
        self.mission_desc.insert("1.0", "High-level mission description and notes.")
        self.mission_desc.pack(fill="both", expand=True)

    def get_state(self) -> dict:
        return {
            "mission_name": self.mission_name.get(),
            "mission_notes": self.mission_notes.get(),
            "central_body": self.central_body.get(),
            "origin_body": self.origin_body.get(),
            "target_body": self.target_body.get(),
            "departure_utc": self.departure_utc.get(),
            "arrival_utc": self.arrival_utc.get(),
            "description": self.mission_desc.get("1.0", "end-1c"),
        }

    def set_state(self, st: dict):
        self.mission_name.set(st.get("mission_name", self.mission_name.get()))
        self.mission_notes.set(st.get("mission_notes", self.mission_notes.get()))
        self.central_body.set(st.get("central_body", self.central_body.get()))
        self.origin_body.set(st.get("origin_body", self.origin_body.get()))
        self.target_body.set(st.get("target_body", self.target_body.get()))
        self.departure_utc.set(st.get("departure_utc", self.departure_utc.get()))
        self.arrival_utc.set(st.get("arrival_utc", self.arrival_utc.get()))
        self.mission_desc.delete("1.0", "end")
        self.mission_desc.insert("1.0", st.get("description", ""))


# ------------------------------------------------------------
# Transfer tab
# ------------------------------------------------------------
class TransferTab(ttk.Frame):
    def __init__(self, parent, app: PMPApp):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        opts = ttk.LabelFrame(outer, text="Lambert Transfer Config")
        opts.pack(fill="x", padx=5, pady=5)

        self.long_way = tk.BooleanVar(value=False)
        self.moi_alt_km = tk.StringVar(value="250")
        self.moi_mu_km3s2 = tk.StringVar(value="4.282837e4")  # Mars μ [km^3/s^2]

        ttk.Checkbutton(opts, text="Use long-way solution", variable=self.long_way).grid(
            row=0, column=0, sticky="w"
        )

        ttk.Label(opts, text="MOI periapsis alt [km]:").grid(row=1, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.moi_alt_km, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(opts, text="Target body μ [km^3/s^2]:").grid(row=1, column=2, sticky="w")
        ttk.Entry(opts, textvariable=self.moi_mu_km3s2, width=14).grid(row=1, column=3, sticky="w")

        for c in range(4):
            opts.columnconfigure(c, weight=1)

        run_frame = ttk.Frame(outer)
        run_frame.pack(fill="x", pady=5)

        ttk.Button(run_frame, text="Run Lambert Transfer", command=self.run_transfer).pack(
            side="left", padx=5
        )

        self.transfer_output = tk.Text(outer, height=18, wrap="word")
        self.transfer_output.pack(fill="both", expand=True, padx=5, pady=5)

    def get_state(self) -> dict:
        return {
            "long_way": self.long_way.get(),
            "moi_alt_km": self.moi_alt_km.get(),
            "moi_mu_km3s2": self.moi_mu_km3s2.get(),
        }

    def set_state(self, st: dict):
        self.long_way.set(st.get("long_way", self.long_way.get()))
        self.moi_alt_km.set(st.get("moi_alt_km", self.moi_alt_km.get()))
        self.moi_mu_km3s2.set(st.get("moi_mu_km3s2", self.moi_mu_km3s2.get()))

    def run_transfer(self):
        if "transfer" in PMP_IMPORT_ERRORS:
            messagebox.showerror("Transfer", f"Transfer module not available:\n{PMP_IMPORT_ERRORS['transfer']}")
            return

        # Lazy import inside function to avoid hard dependency at import time
        from pmp.trajectory.transfer import EndpointConfig, TransferConfig, compute_lambert_transfer
        from poliastro.bodies import Sun, Earth, Mars, Venus, Jupiter
        from astropy import units as u

        # Link mission tab dates
        mstate = self.app.mission_tab.get_state()
        dep_utc = mstate["departure_utc"]
        arr_utc = mstate["arrival_utc"]
        origin_name = mstate["origin_body"].lower()
        target_name = mstate["target_body"].lower()

        # Map simple names to poliastro bodies; user can edit later
        body_map = {
            "earth": Earth,
            "mars": Mars,
            "venus": Venus,
            "jupiter": Jupiter,
        }
        if origin_name not in body_map or target_name not in body_map:
            messagebox.showerror("Transfer", "Origin/target body must be one of: Earth, Mars, Venus, Jupiter.")
            return

        origin = EndpointConfig(epoch_utc=dep_utc, body=body_map[origin_name])
        target = EndpointConfig(epoch_utc=arr_utc, body=body_map[target_name])

        # MOI config
        try:
            moi_alt_km = float(self.moi_alt_km.get())
            moi_mu = float(self.moi_mu_km3s2.get())
        except ValueError:
            moi_alt_km = None
            moi_mu = None

        cfg = TransferConfig(
            central_body=Sun,
            origin=origin,
            target=target,
            long_way=self.long_way.get(),
            moi_body_mu_km3_s2=moi_mu,
            moi_periapsis_alt_km=moi_alt_km,
        )

        try:
            result = compute_lambert_transfer(cfg)
        except Exception as e:
            messagebox.showerror("Transfer Error", str(e))
            return

        # Optional: compute Mars arrival season if target is Mars and module exists
        season_str = ""
        if target_name == "mars" and "seasons" not in PMP_IMPORT_ERRORS:
            try:
                from pmp.seasons.mars_seasons import get_arrival_ls_and_season
                arrival_dt = dt.datetime.fromisoformat(arr_utc)
                tof_days = result.tof_days
                s = get_arrival_ls_and_season(arrival_dt - dt.timedelta(days=tof_days), tof_days)
                season_str = f"\nArrival Ls: {s.ls_deg:.2f} deg\nSeason (north): {s.season_north}"
            except Exception:
                pass

        txt_lines = []
        txt_lines.append(f"Transfer: {origin_name.title()} -> {target_name.title()}")
        txt_lines.append(f"TOF: {result.tof_days:.3f} days")
        txt_lines.append(f"Δv_depart: {result.dv_depart_km_s:.4f} km/s")
        txt_lines.append(f"Δv_arrive: {result.dv_arrive_km_s:.4f} km/s")
        if result.v_inf_arrive_mag_km_s is not None:
            txt_lines.append(f"v_inf_arrive: {result.v_inf_arrive_mag_km_s:.4f} km/s")
        if result.dv_moi_km_s is not None:
            txt_lines.append(f"Approx MOI Δv: {result.dv_moi_km_s:.4f} km/s")
        txt_lines.append(season_str)

        self.transfer_output.delete("1.0", "end")
        self.transfer_output.insert("1.0", "\n".join(txt_lines))

        # Push to summary
        self.app.add_summary_entry("Transfer", {
            "origin": origin_name,
            "target": target_name,
            "tof_days": result.tof_days,
            "dv_depart_km_s": result.dv_depart_km_s,
            "dv_arrive_km_s": result.dv_arrive_km_s,
            "dv_moi_km_s": result.dv_moi_km_s,
        })


# ------------------------------------------------------------
# Probe Env / Power tab
# ------------------------------------------------------------
class ProbeEnvTab(ttk.Frame):
    def __init__(self, parent, app: PMPApp):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        # Environment
        env_frame = ttk.LabelFrame(outer, text="Environment (Probe Surface)")
        env_frame.pack(fill="x", padx=5, pady=5)

        self.lat = tk.StringVar(value="-45.0")
        self.elev = tk.StringVar(value="-3500.0")
        self.tau = tk.StringVar(value="0.4")
        self.Ls_deg = tk.StringVar(value="173.7")
        self.t_mean = tk.StringVar(value="210.0")
        self.t_swing = tk.StringVar(value="60.0")

        ttk.Label(env_frame, text="Lat [deg]:").grid(row=0, column=0, sticky="w")
        ttk.Entry(env_frame, textvariable=self.lat, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(env_frame, text="Elev [m]:").grid(row=0, column=2, sticky="w")
        ttk.Entry(env_frame, textvariable=self.elev, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(env_frame, text="Tau (dust):").grid(row=0, column=4, sticky="w")
        ttk.Entry(env_frame, textvariable=self.tau, width=6).grid(row=0, column=5, sticky="w")

        ttk.Label(env_frame, text="Ls [deg]:").grid(row=1, column=0, sticky="w")
        ttk.Entry(env_frame, textvariable=self.Ls_deg, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(env_frame, text="T_mean [K]:").grid(row=1, column=2, sticky="w")
        ttk.Entry(env_frame, textvariable=self.t_mean, width=8).grid(row=1, column=3, sticky="w")
        ttk.Label(env_frame, text="Daily swing [K]:").grid(row=1, column=4, sticky="w")
        ttk.Entry(env_frame, textvariable=self.t_swing, width=8).grid(row=1, column=5, sticky="w")

        # Power
        pow_frame = ttk.LabelFrame(outer, text="Power & Thermal")
        pow_frame.pack(fill="x", padx=5, pady=5)

        self.sim_hours = tk.StringVar(value="700.0")
        self.dt_s = tk.StringVar(value="30.0")
        self.start_soc = tk.StringVar(value="0.8")

        ttk.Label(pow_frame, text="Sim hours:").grid(row=0, column=0, sticky="w")
        ttk.Entry(pow_frame, textvariable=self.sim_hours, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(pow_frame, text="dt [s]:").grid(row=0, column=2, sticky="w")
        ttk.Entry(pow_frame, textvariable=self.dt_s, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(pow_frame, text="Start SOC [0-1]:").grid(row=0, column=4, sticky="w")
        ttk.Entry(pow_frame, textvariable=self.start_soc, width=8).grid(row=0, column=5, sticky="w")

        self.mli_int = tk.BooleanVar(value=False)
        self.mli_batt = tk.BooleanVar(value=False)
        ttk.Checkbutton(pow_frame, text="MLI Internal", variable=self.mli_int).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(pow_frame, text="MLI Battery", variable=self.mli_batt).grid(row=1, column=2, sticky="w")

        run_frame = ttk.Frame(outer)
        run_frame.pack(fill="x", pady=5)
        ttk.Button(run_frame, text="Run Probe Env/Power Sim", command=self.run_sim).pack(side="left", padx=5)

        self.output = tk.Text(outer, height=16, wrap="word")
        self.output.pack(fill="both", expand=True, padx=5, pady=5)

    def get_state(self) -> dict:
        return {
            "lat": self.lat.get(),
            "elev": self.elev.get(),
            "tau": self.tau.get(),
            "Ls_deg": self.Ls_deg.get(),
            "t_mean": self.t_mean.get(),
            "t_swing": self.t_swing.get(),
            "sim_hours": self.sim_hours.get(),
            "dt_s": self.dt_s.get(),
            "start_soc": self.start_soc.get(),
            "mli_int": self.mli_int.get(),
            "mli_batt": self.mli_batt.get(),
        }

    def set_state(self, st: dict):
        self.lat.set(st.get("lat", self.lat.get()))
        self.elev.set(st.get("elev", self.elev.get()))
        self.tau.set(st.get("tau", self.tau.get()))
        self.Ls_deg.set(st.get("Ls_deg", self.Ls_deg.get()))
        self.t_mean.set(st.get("t_mean", self.t_mean.get()))
        self.t_swing.set(st.get("t_swing", self.t_swing.get()))
        self.sim_hours.set(st.get("sim_hours", self.sim_hours.get()))
        self.dt_s.set(st.get("dt_s", self.dt_s.get()))
        self.start_soc.set(st.get("start_soc", self.start_soc.get()))
        self.mli_int.set(st.get("mli_int", self.mli_int.get()))
        self.mli_batt.set(st.get("mli_batt", self.mli_batt.get()))

    def run_sim(self):
        if "probe_env" in PMP_IMPORT_ERRORS:
            messagebox.showerror("Probe Env/Power", PMP_IMPORT_ERRORS["probe_env"])
            return

        from pmp.power.probe_env import (
            ProbeEnvConfig,
            ProbePanelConfig,
            ProbeBatteryConfig,
            ProbeThermalConfig,
            ProbeEnvSimConfig,
            simulate_probe_env_power,
        )

        try:
            env_cfg = ProbeEnvConfig(
                lat_deg=float(self.lat.get()),
                elev_m=float(self.elev.get()),
                tau=float(self.tau.get()),
                Ls_deg=float(self.Ls_deg.get()),
                t_mean_K=float(self.t_mean.get()),
                daily_swing_K=float(self.t_swing.get()),
            )
            therm_cfg = ProbeThermalConfig(
                mli_internal=self.mli_int.get(),
                mli_batt=self.mli_batt.get(),
            )
            sim_cfg = ProbeEnvSimConfig(
                sim_hours=float(self.sim_hours.get()),
                dt_s=float(self.dt_s.get()),
                env=env_cfg,
                thermal=therm_cfg,
                start_soc=float(self.start_soc.get()),
            )
        except ValueError as e:
            messagebox.showerror("Probe Env/Power", f"Bad input: {e}")
            return

        try:
            res = simulate_probe_env_power(sim_cfg)
        except Exception as e:
            messagebox.showerror("Probe Env/Power", str(e))
            return

        summary = res.summary
        txt_lines = ["Probe Env/Power Simulation Summary:\n"]
        for k, v in summary.items():
            txt_lines.append(f"{k}: {v}")
        txt_lines.append(f"\nRows: {len(res.df)}, Columns: {list(res.df.columns)}")

        self.output.delete("1.0", "end")
        self.output.insert("1.0", "\n".join(txt_lines))

        self.app.add_summary_entry("Probe Env/Power", summary)


# ------------------------------------------------------------
# Orbiter EPS tab
# ------------------------------------------------------------
class OrbiterEPSTab(ttk.Frame):
    def __init__(self, parent, app: PMPApp):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        cfg_frame = ttk.LabelFrame(outer, text="EPS Configuration")
        cfg_frame.pack(fill="x", padx=5, pady=5)

        self.orbit_period = tk.StringVar(value="7200")
        self.eclipse_frac = tk.StringVar(value="0.35")
        self.array_power = tk.StringVar(value="900")
        self.array_inc_eff = tk.StringVar(value="0.9")
        self.batt_cap = tk.StringVar(value="2500")
        self.soc0 = tk.StringVar(value="0.8")
        self.P_sun = tk.StringVar(value="600")
        self.P_ecl = tk.StringVar(value="400")
        self.total_orbits = tk.StringVar(value="30")
        self.dt_s = tk.StringVar(value="10.0")

        row = 0
        ttk.Label(cfg_frame, text="Period [s]:").grid(row=row, column=0, sticky="w")
        ttk.Entry(cfg_frame, textvariable=self.orbit_period, width=10).grid(row=row, column=1, sticky="w")
        ttk.Label(cfg_frame, text="Eclipse frac [0-1]:").grid(row=row, column=2, sticky="w")
        ttk.Entry(cfg_frame, textvariable=self.eclipse_frac, width=8).grid(row=row, column=3, sticky="w")
        row += 1

        ttk.Label(cfg_frame, text="Array power [W]:").grid(row=row, column=0, sticky="w")
        ttk.Entry(cfg_frame, textvariable=self.array_power, width=10).grid(row=row, column=1, sticky="w")
        ttk.Label(cfg_frame, text="Incidence eff:").grid(row=row, column=2, sticky="w")
        ttk.Entry(cfg_frame, textvariable=self.array_inc_eff, width=8).grid(row=row, column=3, sticky="w")
        row += 1

        ttk.Label(cfg_frame, text="Battery cap [Wh]:").grid(row=row, column=0, sticky="w")
        ttk.Entry(cfg_frame, textvariable=self.batt_cap, width=10).grid(row=row, column=1, sticky="w")
        ttk.Label(cfg_frame, text="SOC0 [0-1]:").grid(row=row, column=2, sticky="w")
        ttk.Entry(cfg_frame, textvariable=self.soc0, width=8).grid(row=row, column=3, sticky="w")
        row += 1

        ttk.Label(cfg_frame, text="P_sun [W]:").grid(row=row, column=0, sticky="w")
        ttk.Entry(cfg_frame, textvariable=self.P_sun, width=10).grid(row=row, column=1, sticky="w")
        ttk.Label(cfg_frame, text="P_eclipse [W]:").grid(row=row, column=2, sticky="w")
        ttk.Entry(cfg_frame, textvariable=self.P_ecl, width=10).grid(row=row, column=3, sticky="w")
        row += 1

        ttk.Label(cfg_frame, text="# Orbits:").grid(row=row, column=0, sticky="w")
        ttk.Entry(cfg_frame, textvariable=self.total_orbits, width=8).grid(row=row, column=1, sticky="w")
        ttk.Label(cfg_frame, text="dt [s]:").grid(row=row, column=2, sticky="w")
        ttk.Entry(cfg_frame, textvariable=self.dt_s, width=8).grid(row=row, column=3, sticky="w")

        run_frame = ttk.Frame(outer)
        run_frame.pack(fill="x", pady=5)
        ttk.Button(run_frame, text="Run Orbiter EPS Sim", command=self.run_eps).pack(side="left", padx=5)

        self.output = tk.Text(outer, height=16, wrap="word")
        self.output.pack(fill="both", expand=True, padx=5, pady=5)

    def get_state(self) -> dict:
        return {
            "orbit_period": self.orbit_period.get(),
            "eclipse_frac": self.eclipse_frac.get(),
            "array_power": self.array_power.get(),
            "array_inc_eff": self.array_inc_eff.get(),
            "batt_cap": self.batt_cap.get(),
            "soc0": self.soc0.get(),
            "P_sun": self.P_sun.get(),
            "P_ecl": self.P_ecl.get(),
            "total_orbits": self.total_orbits.get(),
            "dt_s": self.dt_s.get(),
        }

    def set_state(self, st: dict):
        self.orbit_period.set(st.get("orbit_period", self.orbit_period.get()))
        self.eclipse_frac.set(st.get("eclipse_frac", self.eclipse_frac.get()))
        self.array_power.set(st.get("array_power", self.array_power.get()))
        self.array_inc_eff.set(st.get("array_inc_eff", self.array_inc_eff.get()))
        self.batt_cap.set(st.get("batt_cap", self.batt_cap.get()))
        self.soc0.set(st.get("soc0", self.soc0.get()))
        self.P_sun.set(st.get("P_sun", self.P_sun.get()))
        self.P_ecl.set(st.get("P_ecl", self.P_ecl.get()))
        self.total_orbits.set(st.get("total_orbits", self.total_orbits.get()))
        self.dt_s.set(st.get("dt_s", self.dt_s.get()))

    def run_eps(self):
        if "eps" in PMP_IMPORT_ERRORS:
            messagebox.showerror("Orbiter EPS", PMP_IMPORT_ERRORS["eps"])
            return

        from pmp.power.orbiter_eps import (
            OrbitLightingConfig,
            ArrayConfig,
            BatteryConfig,
            LoadConfig,
            OrbiterEPSSimConfig,
            simulate_orbiter_eps,
        )

        try:
            orbit = OrbitLightingConfig(
                period_s=float(self.orbit_period.get()),
                eclipse_fraction=float(self.eclipse_frac.get()),
            )
            array = ArrayConfig(
                nameplate_power_W=float(self.array_power.get()),
                incidence_eff=float(self.array_inc_eff.get()),
            )
            battery = BatteryConfig(
                cap_Wh=float(self.batt_cap.get()),
            )
            loads = LoadConfig(
                P_sun_W=float(self.P_sun.get()),
                P_eclipse_W=float(self.P_ecl.get()),
            )
            cfg = OrbiterEPSSimConfig(
                total_orbits=int(self.total_orbits.get()),
                dt_s=float(self.dt_s.get()),
                orbit=orbit,
                array=array,
                battery=battery,
                loads=loads,
                soc0=float(self.soc0.get()),
            )
        except ValueError as e:
            messagebox.showerror("Orbiter EPS", f"Bad input: {e}")
            return

        try:
            res = simulate_orbiter_eps(cfg)
        except Exception as e:
            messagebox.showerror("Orbiter EPS", str(e))
            return

        meta = res.meta
        orbit_table = res.orbit_table

        txt_lines = ["Orbiter EPS Simulation Summary:\n"]
        txt_lines.append(f"Hit min SOC: {meta['flags']['hit_min_soc_global']}")
        txt_lines.append("\nPer-orbit energy summary (first few rows):")
        txt_lines.append(orbit_table.head().to_string(index=False))

        self.output.delete("1.0", "end")
        self.output.insert("1.0", "\n".join(txt_lines))

        self.app.add_summary_entry("Orbiter EPS", {
            "hit_min_soc": meta["flags"]["hit_min_soc_global"],
            "n_orbits": len(orbit_table),
        })


# ------------------------------------------------------------
# EDL tab
# ------------------------------------------------------------
class EDLTab(ttk.Frame):
    def __init__(self, parent, app: PMPApp):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        body_frame = ttk.LabelFrame(outer, text="Body & Atmosphere")
        body_frame.pack(fill="x", padx=5, pady=5)

        self.mu = tk.StringVar(value="4.282837e13")
        self.radius = tk.StringVar(value="3389500")
        self.rho0 = tk.StringVar(value="0.02")
        self.hscale = tk.StringVar(value="10800")

        ttk.Label(body_frame, text="μ [m^3/s^2]:").grid(row=0, column=0, sticky="w")
        ttk.Entry(body_frame, textvariable=self.mu, width=14).grid(row=0, column=1, sticky="w")
        ttk.Label(body_frame, text="Radius [m]:").grid(row=0, column=2, sticky="w")
        ttk.Entry(body_frame, textvariable=self.radius, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(body_frame, text="rho0 [kg/m³]:").grid(row=1, column=0, sticky="w")
        ttk.Entry(body_frame, textvariable=self.rho0, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(body_frame, text="H [m]:").grid(row=1, column=2, sticky="w")
        ttk.Entry(body_frame, textvariable=self.hscale, width=10).grid(row=1, column=3, sticky="w")

        veh_frame = ttk.LabelFrame(outer, text="Vehicle & Chute")
        veh_frame.pack(fill="x", padx=5, pady=5)

        self.mass = tk.StringVar(value="41.0")
        self.Cd = tk.StringVar(value="1.6")
        self.area = tk.StringVar(value="0.78")
        ttk.Label(veh_frame, text="Mass [kg]:").grid(row=0, column=0, sticky="w")
        ttk.Entry(veh_frame, textvariable=self.mass, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(veh_frame, text="Cd:").grid(row=0, column=2, sticky="w")
        ttk.Entry(veh_frame, textvariable=self.Cd, width=6).grid(row=0, column=3, sticky="w")
        ttk.Label(veh_frame, text="Area [m²]:").grid(row=0, column=4, sticky="w")
        ttk.Entry(veh_frame, textvariable=self.area, width=8).grid(row=0, column=5, sticky="w")

        self.chute_area = tk.StringVar(value="8.0")
        self.chute_cd = tk.StringVar(value="1.6")
        self.chute_deploy_alt = tk.StringVar(value="9000")

        ttk.Label(veh_frame, text="Chute area [m²]:").grid(row=1, column=0, sticky="w")
        ttk.Entry(veh_frame, textvariable=self.chute_area, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(veh_frame, text="Chute Cd:").grid(row=1, column=2, sticky="w")
        ttk.Entry(veh_frame, textvariable=self.chute_cd, width=6).grid(row=1, column=3, sticky="w")
        ttk.Label(veh_frame, text="Chute deploy alt [m]:").grid(row=1, column=4, sticky="w")
        ttk.Entry(veh_frame, textvariable=self.chute_deploy_alt, width=8).grid(row=1, column=5, sticky="w")

        retro_frame = ttk.LabelFrame(outer, text="Retro Burn")
        retro_frame.pack(fill="x", padx=5, pady=5)

        self.retro_thrust = tk.StringVar(value="1400")
        self.retro_isp = tk.StringVar(value="230")
        self.retro_alt = tk.StringVar(value="200")
        self.retro_vcut = tk.StringVar(value="1.0")

        ttk.Label(retro_frame, text="Thrust [N]:").grid(row=0, column=0, sticky="w")
        ttk.Entry(retro_frame, textvariable=self.retro_thrust, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(retro_frame, text="Isp [s]:").grid(row=0, column=2, sticky="w")
        ttk.Entry(retro_frame, textvariable=self.retro_isp, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(retro_frame, text="Burn alt [m]:").grid(row=0, column=4, sticky="w")
        ttk.Entry(retro_frame, textvariable=self.retro_alt, width=8).grid(row=0, column=5, sticky="w")
        ttk.Label(retro_frame, text="Cutoff v [m/s]:").grid(row=0, column=6, sticky="w")
        ttk.Entry(retro_frame, textvariable=self.retro_vcut, width=8).grid(row=0, column=7, sticky="w")

        sim_frame = ttk.LabelFrame(outer, text="Simulation")
        sim_frame.pack(fill="x", padx=5, pady=5)

        self.h0 = tk.StringVar(value="125000")
        self.v0 = tk.StringVar(value="5800")
        self.t_max = tk.StringVar(value="1200")
        self.dt_s = tk.StringVar(value="0.1")
        self.enable_retro = tk.BooleanVar(value=True)

        ttk.Label(sim_frame, text="h0 [m]:").grid(row=0, column=0, sticky="w")
        ttk.Entry(sim_frame, textvariable=self.h0, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(sim_frame, text="v0 [m/s]:").grid(row=0, column=2, sticky="w")
        ttk.Entry(sim_frame, textvariable=self.v0, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(sim_frame, text="t_max [s]:").grid(row=0, column=4, sticky="w")
        ttk.Entry(sim_frame, textvariable=self.t_max, width=8).grid(row=0, column=5, sticky="w")
        ttk.Label(sim_frame, text="dt [s]:").grid(row=0, column=6, sticky="w")
        ttk.Entry(sim_frame, textvariable=self.dt_s, width=6).grid(row=0, column=7, sticky="w")
        ttk.Checkbutton(sim_frame, text="Enable retro", variable=self.enable_retro).grid(row=1, column=0, sticky="w")

        run_frame = ttk.Frame(outer)
        run_frame.pack(fill="x", pady=5)
        ttk.Button(run_frame, text="Run EDL Sim", command=self.run_edl).pack(side="left", padx=5)

        self.output = tk.Text(outer, height=16, wrap="word")
        self.output.pack(fill="both", expand=True, padx=5, pady=5)

    def get_state(self) -> dict:
        return {
            "mu": self.mu.get(),
            "radius": self.radius.get(),
            "rho0": self.rho0.get(),
            "hscale": self.hscale.get(),
            "mass": self.mass.get(),
            "Cd": self.Cd.get(),
            "area": self.area.get(),
            "chute_area": self.chute_area.get(),
            "chute_cd": self.chute_cd.get(),
            "chute_deploy_alt": self.chute_deploy_alt.get(),
            "retro_thrust": self.retro_thrust.get(),
            "retro_isp": self.retro_isp.get(),
            "retro_alt": self.retro_alt.get(),
            "retro_vcut": self.retro_vcut.get(),
            "h0": self.h0.get(),
            "v0": self.v0.get(),
            "t_max": self.t_max.get(),
            "dt_s": self.dt_s.get(),
            "enable_retro": self.enable_retro.get(),
        }

    def set_state(self, st: dict):
        for k, var in [
            ("mu", self.mu),
            ("radius", self.radius),
            ("rho0", self.rho0),
            ("hscale", self.hscale),
            ("mass", self.mass),
            ("Cd", self.Cd),
            ("area", self.area),
            ("chute_area", self.chute_area),
            ("chute_cd", self.chute_cd),
            ("chute_deploy_alt", self.chute_deploy_alt),
            ("retro_thrust", self.retro_thrust),
            ("retro_isp", self.retro_isp),
            ("retro_alt", self.retro_alt),
            ("retro_vcut", self.retro_vcut),
            ("h0", self.h0),
            ("v0", self.v0),
            ("t_max", self.t_max),
            ("dt_s", self.dt_s),
        ]:
            var.set(st.get(k, var.get()))
        self.enable_retro.set(st.get("enable_retro", self.enable_retro.get()))

    def run_edl(self):
        if "edl" in PMP_IMPORT_ERRORS:
            messagebox.showerror("EDL", PMP_IMPORT_ERRORS["edl"])
            return

        from pmp.edl.edl_1d import (
            BodyAtmosphere,
            VehicleAeroshell,
            ParachuteConfig,
            RetroConfig,
            EDLSimConfig,
            simulate_edl,
        )

        try:
            body = BodyAtmosphere(
                mu_m3_s2=float(self.mu.get()),
                radius_m=float(self.radius.get()),
                rho0_kg_m3=float(self.rho0.get()),
                hscale_m=float(self.hscale.get()),
            )
            veh = VehicleAeroshell(
                mass_kg=float(self.mass.get()),
                Cd=float(self.Cd.get()),
                area_m2=float(self.area.get()),
            )
            chute = ParachuteConfig(
                Cd=float(self.chute_cd.get()),
                area_m2=float(self.chute_area.get()),
                deploy_alt_m=float(self.chute_deploy_alt.get()),
            )
            retro = RetroConfig(
                thrust_N=float(self.retro_thrust.get()),
                Isp_s=float(self.retro_isp.get()),
                burn_alt_m=float(self.retro_alt.get()),
                cutoff_v_mps=float(self.retro_vcut.get()),
            )
            cfg = EDLSimConfig(
                body=body,
                vehicle=veh,
                t_max_s=float(self.t_max.get()),
                dt_s=float(self.dt_s.get()),
                h0_m=float(self.h0.get()),
                v0_mps=float(self.v0.get()),
                parachute=chute,
                retro=retro,
                enable_retro=self.enable_retro.get(),
            )
        except ValueError as e:
            messagebox.showerror("EDL", f"Bad input: {e}")
            return

        try:
            res = simulate_edl(cfg)
        except Exception as e:
            messagebox.showerror("EDL", str(e))
            return

        # Summarize results
        touchdown = [ev for ev in res.events if ev.label == "TOUCHDOWN"]
        if touchdown:
            info = touchdown[-1].details
            v_imp = info["impact_v_mps"]
            mass = info["mass_kg"]
        else:
            v_imp = None
            mass = None

        txt_lines = ["EDL Simulation Summary:\n"]
        if v_imp is not None:
            txt_lines.append(f"Touchdown speed: {v_imp:.2f} m/s")
            txt_lines.append(f"Final mass: {mass:.2f} kg")
        else:
            txt_lines.append("No touchdown event within t_max.")

        txt_lines.append("\nEvents:")
        for ev in res.events:
            txt_lines.append(f"{ev.t_s:.2f}s: {ev.label} {ev.details}")

        self.output.delete("1.0", "end")
        self.output.insert("1.0", "\n".join(txt_lines))

        self.app.add_summary_entry("EDL", {
            "impact_v_mps": v_imp,
            "events": [e.label for e in res.events],
        })


# ------------------------------------------------------------
# Comms tab
# ------------------------------------------------------------
class CommsTab(ttk.Frame):
    def __init__(self, parent, app: PMPApp):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        # Probe–orbiter link
        u_frame = ttk.LabelFrame(outer, text="Probe–Orbiter UHF/LoRa")
        u_frame.pack(fill="x", padx=5, pady=5)

        self.range_km = tk.StringVar(value="250")
        self.bitrate = tk.StringVar(value="5470")
        self.pass_dur = tk.StringVar(value="600")

        ttk.Label(u_frame, text="Range [km]:").grid(row=0, column=0, sticky="w")
        ttk.Entry(u_frame, textvariable=self.range_km, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(u_frame, text="Bitrate [bps]:").grid(row=0, column=2, sticky="w")
        ttk.Entry(u_frame, textvariable=self.bitrate, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(u_frame, text="Pass duration [s]:").grid(row=0, column=4, sticky="w")
        ttk.Entry(u_frame, textvariable=self.pass_dur, width=8).grid(row=0, column=5, sticky="w")

        # Deep-space link
        d_frame = ttk.LabelFrame(outer, text="Deep-Space X-band")
        d_frame.pack(fill="x", padx=5, pady=5)

        self.range_ds_km = tk.StringVar(value="2.5e8")
        ttk.Label(d_frame, text="Range [km]:").grid(row=0, column=0, sticky="w")
        ttk.Entry(d_frame, textvariable=self.range_ds_km, width=10).grid(row=0, column=1, sticky="w")

        run_frame = ttk.Frame(outer)
        run_frame.pack(fill="x", pady=5)

        ttk.Button(run_frame, text="Run Comms Check", command=self.run_comms).pack(side="left", padx=5)

        self.output = tk.Text(outer, height=18, wrap="word")
        self.output.pack(fill="both", expand=True, padx=5, pady=5)

    def get_state(self) -> dict:
        return {
            "range_km": self.range_km.get(),
            "bitrate": self.bitrate.get(),
            "pass_dur": self.pass_dur.get(),
            "range_ds_km": self.range_ds_km.get(),
        }

    def set_state(self, st: dict):
        self.range_km.set(st.get("range_km", self.range_km.get()))
        self.bitrate.set(st.get("bitrate", self.bitrate.get()))
        self.pass_dur.set(st.get("pass_dur", self.pass_dur.get()))
        self.range_ds_km.set(st.get("range_ds_km", self.range_ds_km.get()))

    def run_comms(self):
        if "comms" in PMP_IMPORT_ERRORS:
            messagebox.showerror("Comms", PMP_IMPORT_ERRORS["comms"])
            return

        from pmp.comms.link_mars import (
            magpie_default_probe_915,
            magpie_default_orbiter_radio,
            StaticPassConfig,
            compute_static_probe_orbiter_link,
            magpie_default_deep_space_cfg,
            compute_deep_space_link,
        )

        try:
            range_km = float(self.range_km.get())
            bitrate = float(self.bitrate.get())
            dur_s = float(self.pass_dur.get())
            range_ds = float(self.range_ds_km.get())
        except ValueError as e:
            messagebox.showerror("Comms", f"Bad input: {e}")
            return

        probe = magpie_default_probe_915()
        orb = magpie_default_orbiter_radio()
        static_cfg = StaticPassConfig(bitrate_bps=bitrate, duration_s=dur_s)

        try:
            static_res = compute_static_probe_orbiter_link(probe, orb, range_km, static_cfg)
            ds_cfg = magpie_default_deep_space_cfg()
            ds_res = compute_deep_space_link(range_km=range_ds, cfg=ds_cfg)
        except Exception as e:
            messagebox.showerror("Comms", str(e))
            return

        txt_lines = ["Probe–Orbiter Link:"]
        txt_lines.append(f"Range: {static_res.range_km:.1f} km")
        txt_lines.append(f"Uplink margin: {static_res.uplink_margin_db:.2f} dB")
        txt_lines.append(f"Downlink margin: {static_res.downlink_margin_db:.2f} dB")
        txt_lines.append(f"Static data: {static_res.static_data_MB:.3f} MB")

        txt_lines.append("\nDeep-Space X-band Link:")
        txt_lines.append(f"Range: {ds_res.range_km:.3e} km")
        txt_lines.append(f"Link margin: {ds_res.link_margin_db:.2f} dB")
        txt_lines.append(f"Eb/N0 margin: {ds_res.ebn0_margin_db:.2f} dB")
        txt_lines.append(f"Rx power: {ds_res.rx_power_dbm:.2f} dBm")

        self.output.delete("1.0", "end")
        self.output.insert("1.0", "\n".join(txt_lines))

        self.app.add_summary_entry("Comms", {
            "uplink_margin_db": static_res.uplink_margin_db,
            "downlink_margin_db": static_res.downlink_margin_db,
            "static_data_MB": static_res.static_data_MB,
            "ds_link_margin_db": ds_res.link_margin_db,
            "ds_ebn0_margin_db": ds_res.ebn0_margin_db,
        })


# ------------------------------------------------------------
# ADCS tab
# ------------------------------------------------------------
class ADCSTab(ttk.Frame):
    def __init__(self, parent, app: PMPApp):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        config_frame = ttk.LabelFrame(outer, text="Slew Scenario")
        config_frame.pack(fill="x", padx=5, pady=5)

        self.Kp = tk.StringVar(value="0.2")
        self.Kd = tk.StringVar(value="25.0")
        self.angle_deg = tk.StringVar(value="60.0")
        self.axis = tk.StringVar(value="0 1 0")
        self.t_final = tk.StringVar(value="2000")
        self.dt_s = tk.StringVar(value="0.5")

        ttk.Label(config_frame, text="Kp:").grid(row=0, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.Kp, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(config_frame, text="Kd:").grid(row=0, column=2, sticky="w")
        ttk.Entry(config_frame, textvariable=self.Kd, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(config_frame, text="Angle [deg]:").grid(row=1, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.angle_deg, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(config_frame, text="Axis (x y z):").grid(row=1, column=2, sticky="w")
        ttk.Entry(config_frame, textvariable=self.axis, width=10).grid(row=1, column=3, sticky="w")

        ttk.Label(config_frame, text="t_final [s]:").grid(row=2, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.t_final, width=8).grid(row=2, column=1, sticky="w")
        ttk.Label(config_frame, text="dt [s]:").grid(row=2, column=2, sticky="w")
        ttk.Entry(config_frame, textvariable=self.dt_s, width=8).grid(row=2, column=3, sticky="w")

        run_frame = ttk.Frame(outer)
        run_frame.pack(fill="x", pady=5)
        ttk.Button(run_frame, text="Run Orbiter Slew (MAGPIE defaults)", command=self.run_slew).pack(
            side="left", padx=5
        )

        self.output = tk.Text(outer, height=18, wrap="word")
        self.output.pack(fill="both", expand=True, padx=5, pady=5)

    def get_state(self) -> dict:
        return {
            "Kp": self.Kp.get(),
            "Kd": self.Kd.get(),
            "angle_deg": self.angle_deg.get(),
            "axis": self.axis.get(),
            "t_final": self.t_final.get(),
            "dt_s": self.dt_s.get(),
        }

    def set_state(self, st: dict):
        self.Kp.set(st.get("Kp", self.Kp.get()))
        self.Kd.set(st.get("Kd", self.Kd.get()))
        self.angle_deg.set(st.get("angle_deg", self.angle_deg.get()))
        self.axis.set(st.get("axis", self.axis.get()))
        self.t_final.set(st.get("t_final", self.t_final.get()))
        self.dt_s.set(st.get("dt_s", self.dt_s.get()))

    def run_slew(self):
        if "adcs" in PMP_IMPORT_ERRORS:
            messagebox.showerror("ADCS", PMP_IMPORT_ERRORS["adcs"])
            return

        from pmp.trajectory.adcs import (
            magpie_default_orbiter_body,
            magpie_default_orbiter_rw_cluster,
            SlewPDConfig,
            ADCSSlewSimConfig,
            simulate_pd_slew,
        )

        try:
            Kp = float(self.Kp.get())
            Kd = float(self.Kd.get())
            angle_deg = float(self.angle_deg.get())
            axis_vec = [float(x) for x in self.axis.get().split()]
            t_final = float(self.t_final.get())
            dt_s = float(self.dt_s.get())
        except ValueError as e:
            messagebox.showerror("ADCS", f"Bad input: {e}")
            return

        import numpy as np

        body = magpie_default_orbiter_body()
        rw = magpie_default_orbiter_rw_cluster()
        ctrl = SlewPDConfig(
            Kp=Kp,
            Kd=Kd,
            axis_body=np.array(axis_vec),
            angle_deg=angle_deg,
            t_final=t_final,
            dt=dt_s,
        )
        cfg = ADCSSlewSimConfig(body=body, rw=rw, control=ctrl)

        try:
            res = simulate_pd_slew(cfg)
        except Exception as e:
            messagebox.showerror("ADCS", str(e))
            return

        txt_lines = ["ADCS Slew Result:\n"]
        for k, v in res.metrics.items():
            txt_lines.append(f"{k}: {v:.6g}")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", "\n".join(txt_lines))

        self.app.add_summary_entry("ADCS", res.metrics)


# ------------------------------------------------------------
# Summary tab
# ------------------------------------------------------------
class SummaryTab(ttk.Frame):
    def __init__(self, parent, app: PMPApp):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=10)
        self.text = tk.Text(outer, wrap="word")
        self.text.pack(fill="both", expand=True)

    def refresh(self):
        self.text.delete("1.0", "end")
        if not self.app.summary_entries:
            self.text.insert("1.0", "No results yet. Run subsystems to populate summary.")
            return
        lines = []
        for title, data in self.app.summary_entries[-50:]:  # last up to 50 entries
            lines.append(f"=== {title} ===")
            for k, v in data.items():
                lines.append(f"{k}: {v}")
            lines.append("")
        self.text.insert("1.0", "\n".join(lines))


def main():
    app = PMPApp()
    app.mainloop()


if __name__ == "__main__":
    main()
