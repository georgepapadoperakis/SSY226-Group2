# Python: γ networks for the control-oriented model (COM)

The NMPC predicts 30 s ahead with a control-oriented model (COM) of the car's thermal system. The COM has tuning
parameters γ that the paper looks up in an interpolation table. Here a small neural network computes them instead,
from the current state, the ambient temperature and the planned inputs. Two versions:

| Model | COM | Tuning parameters | File (Python / MATLAB) |
|---|---|---|---|
| **nn11** | original COM (as in the paper) | 11 γ | `models/gamma11_original_com.pt` / `.mat` |
| **nn15** | fixed COM (heat-exchanger, cabin and heater fixes, no extra states) | 15 γ | `models/gamma15_fixed_com.pt` / `.mat` |
| table | original COM | 11 γ from the paper's interpolation table | `../Data/tunedVariables_minus5_minus10.mat` |

The networks are trained end-to-end through the COM (PyTorch) on rule-based simulations of the Simscape model (HFM).

## Setup

```
conda env create -f ../environment.yml      # once
conda activate ssy226
cd python
```

## Quick start (about 5 min)

```
python test_com_matlab_vs_python.py   # 1. the PyTorch COM matches MATLAB (original and fixed COM)   -> ALL PASS
python test_models_open_loop.py       # 2. table / nn11 / nn15 on the 8 test runs                  (~3 min)
python make_figures_open_loop.py      # 3. LaTeX figures and tables of test 2  -> figures/open_loop/
python make_figures_nmpc.py           # 4. closed-loop figures from MATLAB runs -> figures/nmpc/   (needs ../Results)
```

Expected result of step 2 (8 test runs, none used in training):

| | 30-s horizon RMSE: components | cabin air | p_out | 1-h free run MAE: components | cabin air | p_out |
|---|---|---|---|---|---|---|
| Table, original COM | 0.37 K | 0.61 K | 4.44 bar | 2.31 °C | 3.34 °C | 4.35 bar |
| NN 11 γ, original COM | 0.30 K | 0.31 K | 0.93 bar | 2.72 °C | 1.13 °C | 1.97 bar |
| NN 15 γ, fixed COM | **0.16 K** | **0.31 K** | **0.91 bar** | **0.33 °C** | **0.87 °C** | **0.95 bar** |

Components = mean over motor, inverter, DC-DC and battery temperature.

## Data

* **Drive cycles:** WLTCHW (the paper's cycle), NEDC, UDDS, US06; ambient 0, −5, −7, −10, −12, −15, −20, −25, −30 °C.
  All 36 runs use the rule-based controller and last 1 h.
* **Training:** WLTCHW, NEDC, US06 at 0/−5/−10/−15/−20/−30 °C (18 runs).
* **Test (never used in training):** UDDS at 0/−7/−12/−20/−25 °C, WLTCHW −7 °C, NEDC −12 °C, US06 −7 °C.
* `data_cache/<CYCLE>_<T>.pt`: each run converted to COM units (states, inputs, heat loads, fluid properties),
  e.g. `UDDS_minus7C.pt` = UDDS at −7 °C. These are committed, so nothing needs MATLAB to run.
  They are rebuilt from the MATLAB exports (`../Data/tuning_data_*.mat`, `../Results/run_*/training_data_rule_*.mat`)
  with `gamma_net.load_all_profiles` when those are newer.
* `data_cache/fluid_property_table.pt`: CoolProp lookup tables (rebuilt automatically in ~1 min if deleted).

## Files

| File | What it is |
|---|---|
| `com_torch.py` | The COM in PyTorch (port of `Source Files/dynamics_mexfreeNew.m`). `set_fixes(True/False)` switches fixed / original COM. |
| `gamma_net.py` | Library: data loading, train/test split, 30-s windows, COM rollout, the interpolation table, the network (`GammaNet`) and its inputs (`features`). |
| `gamma_freerun.py` | Library: free-run training, the 12/15-output networks, `load_gamma11()`, `load_gamma15()`. |
| `fluid_props_coolprop.py`, `fluid_props_table.py` | Refrigerant/air properties: CoolProp (port of `estimateParameters.m`) and a fast differentiable lookup table. |
| `test_com_matlab_vs_python.py` | Test 1: PyTorch COM vs MATLAB on `test_data/com_testcases_*.mat`. |
| `test_models_open_loop.py` | Test 2: table / nn11 / nn15, 30-s horizon and 1-h free run → `results/open_loop_test_results.pt`. |
| `make_figures_open_loop.py` | LaTeX figures + tables of test 2 → `figures/open_loop/`. |
| `make_figures_nmpc.py` | LaTeX figures + table of the closed-loop MATLAB runs (newest run of each model in `../Results`) → `figures/nmpc/`. |
| `export_networks_to_matlab.py` | `.pt` → `.mat` for MATLAB (`Source Files/gamma_network.m`), with self-test cases. |
| `train_gamma11_step1_base.py` | Training 11 γ, step 1 (~2 h): base network on 3 cycles → `models/history/gamma11_step1_base.pt`. |
| `train_gamma11_step2_startup.py` | Training 11 γ, step 2 (~25 min): free-run start-up fine-tuning → `models/gamma11_original_com.pt`. |
| `train_gamma15_step1_gamma12.py` | Training 15 γ, step 1 (~40 min): adds γ12 (suction density) → `models/history/gamma12_step1_suction_density.pt`. |
| `train_gamma15_step2_fixed_com.py` | Training 15 γ, step 2 (~60 min): fixed COM + γ13–γ15 → `models/gamma15_fixed_com.pt`. |
| `notebook_gamma_network.ipynb` | The development notebook (first WLTCHW-only network, multi-cycle experiment). |
| `report_cycles_experiment.py` | Tables/figures for section 6 of the notebook. |
| `models/history/` | Intermediate networks of the training chain (needed to retrain), incl. the first WLTCHW-only network. |
| `results/` | Outputs of the tests and the training evaluation. |

## Retraining

The training scripts overwrite the files in `models/`. After retraining, run `python export_networks_to_matlab.py`
so MATLAB uses the new weights, and check with `gamma_network()` in MATLAB (prints PASS).
Order: `train_gamma11_step1_base` → `train_gamma11_step2_startup` → `train_gamma15_step1_gamma12` → `train_gamma15_step2_fixed_com`.

## Using the networks in MATLAB (closed loop)

See the main `README.md`: `RunAllComModels` runs the rule-based controller and the NMPC with table, nn11 and nn15,
then `python make_figures_nmpc.py` plots them together.
