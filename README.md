# Energy-Optimal Thermal Management of Heat-Pump Battery Electric Vehicles

[![MATLAB](https://img.shields.io/badge/MATLAB-R2024b%2B-blue)](https://www.mathworks.com/products/matlab.html)
[![CasADi](https://img.shields.io/badge/CasADi-3.6%2B-green)](https://web.casadi.org/)
[![CoolProp](https://img.shields.io/badge/CoolProp-6.x-orange)](http://www.coolprop.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

This repository contains the MATLAB/CasADi implementation accompanying:

> **P. Lokur and N. Murgovski**, "Energy-Optimal Thermal Management of Heat-Pump Battery Electric Vehicles," submitted to *IEEE Open Journal of Vehicular Technology*, 2026.

The framework coordinates six thermal actuators (compressor, cabin blower, motor pump, battery pump, auxiliary heater, radiator fan) via constrained Nonlinear Model Predictive Control (NMPC) to minimize thermal energy consumption while maintaining cabin comfort and component temperature limits.

## SSY226 group project: neural-network tuning parameters for the COM

The NMPC predicts with a control-oriented model (COM) whose tuning parameters γ the paper looks up in an
interpolation table. This branch adds a neural network that computes γ instead, plus a fixed COM version.
The NMPC can run with three prediction models:

| `comModel` | COM | Tuning parameters |
|---|---|---|
| `'table'` | original (paper) | 11 γ from the interpolation table `prepParaVec` (the paper's setup) |
| `'nn11'` | original (paper) | 11 γ from the network `python/models/gamma11_original_com.mat` |
| `'nn15'` | fixed (heat-exchanger, cabin and heater fixes, no extra states) | 15 γ from `python/models/gamma15_fixed_com.mat` |

**Closed loop in MATLAB R2024b** (from the repo root):

```matlab
RunAllComModels                                         % WLTCHW -10 °C: rule + NMPC table/nn11/nn15 (~1 h 45 min)
RunAllComModels(-7, 'UDDS')                             % other temperature / drive cycle
RunThermalComparison('nmpc', -10, 'WLTCHW', false, 'nn15')   % one NMPC run with one model (~30 min)
```

Each run gets a folder named after what was run, e.g. `Results/run_<time>_WLTCHW_rule`,
`Results/run_<time>_WLTCHW_nmpc-nn15`. Then, in the `python` folder: `python make_figures_nmpc.py`
(energy, cabin, battery and inputs of all runs in one figure, plus a LaTeX table → `python/figures/nmpc/`).
The default model for plain `RunThermalComparison` calls is the `COM_MODEL = ...` line in `init_nmpc.m`.

**Open loop and training in Python:** see [`python/README.md`](python/README.md)
(tests of both networks on 8 held-out runs, figures, training scripts).

**Solver:** `init_nmpc.m` uses MA97 when a CoinHSL licence copy is found next to the repo
(`../CoinHSL/` and `../casadi-3.7.1-hsl/`, personal licence, never commit it), otherwise MUMPS. Results are the
same; MUMPS is slower.

**Results so far (WLTCHW, −10 °C, 1 h):** rule-based 3019 Wh (paper: 3020). NMPC with the table −26.2 %
(paper: −27.8 %), with nn11 −3.0 %, with nn15 −11.5 % (`python/figures/nmpc/`). Both networks predict better than the table on
held-out data (open loop), but they were trained on rule-based data only; under NMPC inputs their errors
grow (heater stays on, battery too warm). Retraining with closed-loop NMPC data is the next step.

---

## Requirements

| Package | Version | Notes |
|---------|---------|-------|
| MATLAB | R2024b+ | With Simulink and Simscape |
| [CasADi](https://web.casadi.org/get/) | 3.6+ | IPOPT bundled |
| [CoolProp](http://www.coolprop.org/) | 6.x | Python package, called via MATLAB's `py` interface |
| HSL MA97 | optional | Faster linear solver; MUMPS is used by default |

**Plant model:** [MathWorks Simscape Electric Vehicle Thermal Management with Heat Pump](https://www.mathworks.com/help/simscape-fluids/ug/electric-vehicle-thermal-management-heat-pump.html) (included with Simscape Fluids).

---

## Setup

1. **Install dependencies** listed above.

2. **Add paths** in MATLAB:
   ```matlab
   addpath(genpath('<path_to_casadi>'));
   addpath(genpath('<path_to_this_repo>'));
   ```

3. **Verify CoolProp:**
   ```matlab
   py.CoolProp.CoolProp.PropsSI('T', 'P', 2e5, 'Q', 1, 'R134a')
   ```

4. **Verify CasADi:**
   ```matlab
   import casadi.*
   casadi.SX.sym('test');
   ```

> **Note:** The first simulation step takes few seconds to minutes to build the CasADi symbolic graph and compile the IPOPT solver. Subsequent steps reuse the persistent solver object.

---

## Usage

```matlab
% Run NMPC vs rule-based comparison at multiple ambient temperatures:
RunThermalComparison('both', [-5, -7, -10])

% NMPC only at a single temperature:
RunThermalComparison('nmpc', -10)
```

### Drive cycles

Every run uses one drive cycle, chosen by its handle (list in `Source Files/drive_cycle_list.m`):

| Handle | Cycle |
|---|---|
| `WLTCHW` | Original: WLTC + 130 km/h highway (paper, all data before 18 Nov). Default. |
| `NEDC` | 4 × urban ECE-15 + extra-urban EUDC |
| `UDDS` | EPA city stop-and-go |
| `US06` | EPA aggressive, hard accelerations (stress test) |

Each cycle is repeated to 3600 s (one-hour cold start). The cycles are stored as Signal Editor scenarios in
`DriveCycles_SSY226.mat`, built automatically by `make_drive_cycles` from `Data/drive_cycles/`. The Scenario
block is switched in memory for the run and restored afterwards; the `.slx` file is not modified.

```matlab
RunThermalComparison                                 % menu: controller, temperatures, drive cycles
RunThermalComparison('rule', [-10 -7], 'UDDS')       % one cycle
RunThermalComparison('rule', 'all', 'all', false)    % all 9 temperatures on all cycles, no simlog files

RunDriveCycleDataset('DryRun', true)                 % show the dataset plan
RunDriveCycleDataset                                 % NEDC, UDDS, US06 at all 9 temperatures; skips runs
                                                     % already exported, checks every export, prints a summary
```

Output: `Results/run_<time>_<handle>/training_data_<ctrl>_T<|T|>_<handle>.mat` (`trainingData.cycle` holds
the handle). In Python, `gamma_net.load_all_profiles(..., cycles=("UDDS",))` loads them with keys like `"UDDS_12"`;
section 6 of `python/notebook_gamma_network.ipynb` tests the γ network on them.

To reset the solver (required when changing horizon, sample time, bounds, or model structure):
```matlab
clear MPCController_step_coreLQRNew
delete('nmpc_warm_cache_v4.mat')
```

---

## Repository Structure

| File | Description |
|------|-------------|
| `RunThermalComparison.m` | Entry point: multi-scenario NMPC vs rule-based comparison |
| `RunAllComModels.m` | Rule-based + NMPC with all three prediction models (table, nn11, nn15) |
| `init_nmpc.m` | NMPC settings: horizon, solver (MA97/MUMPS), prediction model `COM_MODEL` |
| `MPCController_stepNew.m` | Simulink wrapper: unpacks bus, unit conversions, calls core |
| `MPCController_step_coreLQRNew.m` | NMPC core: NLP construction, IPOPT solve, warm-starting |
| `create_dynamics_nmpcNew.m` | CasADi symbolic dynamics `F(x, u, p)` |
| `dynamics_mexfreeNew.m` | Control-oriented thermal model (CasADi-compatible) |
| `estimateParameters.m` | CoolProp fluid property estimation at each step |
| `thermalLogicAndEstimation.m` | Supervisory layer: mode selection, EXV commands |
| `prepParaVec.m` | Parameter scheduling via interpolation (the `'table'` model) |
| `gamma_network.m` | Tuning parameters from the γ network (`'nn11'`, `'nn15'`); `gamma_network()` = self-test |
| `export_com_testcases.m` | Test cases for the Python COM port (`python/test_data/`) |
| `export_heat_balance.m` | Exports the HFM heat flows of a run (used for the COM fault analysis) |
| `export_training_data.m` | 1-Hz training data of a run (`training_data_*.mat`) |
| `python/` | γ networks, PyTorch COM, tests and figures (see `python/README.md`) |

---

## NMPC Overview

- **Solver:** IPOPT with exact Hessian
- **Linear solver:** MA97 
- **Horizon:** N = 30 steps, Δt = 1 s
- **States (9):** Motor, inverter, DC-DC temperatures; SOC; battery temperature; refrigerant pressures (low/high); cabin interior and air temperatures
- **Inputs (6):** Compressor speed, blower flow, motor pump speed, battery pump speed, heater power, radiator fan speed
- **Terminal cost:** DARE-based with 4-level fallback (discounted DARE → finite Riccati → Lyapunov → scaled identity)
- **Soft constraints:** Initial condition, ramp-rate, preferred state bounds, algebraic constraints (pressure ratio, superheat, coolant limits)

---

## Citation

If you use this code, please cite the accompanying paper:

> P. Lokur and N. Murgovski, "Energy-Optimal Thermal Management of Heat-Pump Battery Electric Vehicles," submitted to *IEEE Open Journal of Vehicular Technology*, 2026.

A BibTeX entry will be provided upon publication.

## License

MIT License — see [LICENSE](LICENSE).
