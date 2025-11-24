# Piper Mission Planner (PMP)
A modular, mission-agnostic Python framework for full-duration spacecraft mission simulation.  
PMP is designed to support any mission—from Earth-Mars transfers, to deep-space relays, to multi-probe networks.

Originally developed from the MAGPIE Mars probe project.

---

## Features

### Trajectory & Mission Design
- Multi-body Lambert solver (any central body, any endpoints)
- Arrival v∞, Δv depart/arrive, TOF, and optional MOI Δv
- Clean API for chaining mission legs (future)

### 🛰️ ADCS & Slew Dynamics
- Reaction-wheel cluster simulation
- Rigid-body dynamics
- PD slew controller
- RW energy usage & power estimation
- Preset MAGPIE orbiter/probe configurations

### Orbiter EPS
- Solar array generation model
- Eclipse/lighting modeling
- Battery SOC tracking
- Time-series DataFrame + per-orbit energy table
- MAGPIE default EPS configuration

###  Probe Environment & Power
- Thermal conduction, radiation, and sky models
- Battery discharge/charge
- Solar power model
- Diurnal surface temperature cycle
- Long-duration (hundreds of hours) survival simulation

###  Entry-Descent-Landing (EDL)
- Generic 1D vertical EDL kernel
- Exponential atmosphere
- Parachute deployment + corridor logic
- Retropropulsive landing
- Event timeline (deploy, retro, touchdown)

###  Communications
- Probe–orbiter UHF/LoRa link budget
- Static pass data estimation
- Deep-space X-band (DSN-style)
- Link margins, Eb/N0, data volume

###  Seasons & Environment
- Mars Ls and season determination (via marstime)
- Expandable to other bodies

###  GUI (Tkinter)
- Portable, no external dependencies
- Tabs for each subsystem
- Run simulations interactively
- Save/load mission configs (JSON)

###  MAGPIE Mission Orchestrator
A full end-to-end mission chain:
1. Earth → Mars transfer  
2. EDL  
3. Probe surface env/power  
4. Orbiter EPS  
5. Probe ↔ orbiter comm  
6. DSN deep-space comm

---

## Installation

```bash
git clone https://github.com/piperliam/pmp.git
cd pmp
pip install -e .
