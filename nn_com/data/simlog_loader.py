"""Unified self-contained loader for exported Simscape runs and Abdiraham trainingData files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio


@dataclass
class ScenarioData:
    """Standardized 1 Hz trajectory dataset for open-loop COM simulation & training."""

    name: str
    cycle: str
    tamb_c: float
    source_path: Path
    source_format: str
    t: np.ndarray              # [N] time in seconds
    X: np.ndarray              # [N, 9] states [motT, invT, dcdcT, batSoc, batT, p_in, p_out, cabIntT, cabAirT] (K, -, Pa)
    U: np.ndarray              # [N, 6] inputs [compNrpm, cabAirMfIn, motPumpNrpm, batPumpNrpm, heaterPwr, radFanrpm]
    motQ: np.ndarray           # [N] W
    invQ: np.ndarray           # [N] W
    dcdcQ: np.ndarray          # [N] W
    batI: np.ndarray           # [N] A
    speed_kmh: np.ndarray      # [N] km/h
    clntInvTout_K: np.ndarray  # [N] K
    clntBatTin_K: np.ndarray   # [N] K
    batI_avg: np.ndarray       # [N] 120s causal moving average of batI (A)


def causal_moving_avg(x: np.ndarray, window: int = 120) -> np.ndarray:
    """Compute causal moving average matching get_tuneS.m's 120-sample ring buffer."""
    x = np.asarray(x, dtype=np.float64).ravel()
    csum = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - window + 1)
    return (csum[idx + 1] - csum[lo]) / (idx - lo + 1)


def _load_baseline_speed(repo_root: Path, t_grid: np.ndarray) -> np.ndarray:
    """Load default WLTC_HWY speed profile (km/h) from plain numeric MAT or trainingData."""
    plain_mat = repo_root / "Data" / "drive_cycles" / "WLTC_HWY_speed.mat"
    if plain_mat.is_file():
        m = sio.loadmat(plain_mat, squeeze_me=True)
        tv = np.asarray(m["time"], dtype=np.float64).ravel()
        dv = np.asarray(m["speed_kmh"], dtype=np.float64).ravel()
        tv_u, idx_u = np.unique(tv, return_index=True)
        return np.interp(t_grid, tv_u, dv[idx_u], left=0.0, right=0.0)

    # Fallback: read vehSpeed_kmh from any WLTC_HWY training_data_rule_T*.mat in Results/
    results_dir = repo_root / "Results"
    if results_dir.is_dir():
        for cand in sorted(results_dir.rglob("training_data_rule_T*.mat")):
            if any(cyc in cand.stem.upper() for cyc in ("NEDC", "UDDS", "US06")):
                continue
            m = sio.loadmat(cand, squeeze_me=True, struct_as_record=False)
            td = m.get("trainingData", None)
            if td is not None and hasattr(td, "vehSpeed_kmh"):
                tv = np.asarray(td.t_s, dtype=np.float64).ravel()
                dv = np.asarray(td.vehSpeed_kmh, dtype=np.float64).ravel()
                tv_u, idx_u = np.unique(tv, return_index=True)
                return np.interp(t_grid, tv_u, dv[idx_u], left=0.0, right=0.0)

    raise FileNotFoundError(
        "Cannot decode DriveCycle timeseries from ElectricVehicleThermalManagementWithHeatPumpVehicleSpeed.mat "
        "and no fallback WLTC_HWY speed found in Data/drive_cycles/ or Results/."
    )


def _infer_cycle_from_name(path: Path) -> str:
    stem = path.stem.upper()
    for cyc in ("NEDC", "UDDS", "US06", "WLTC"):
        if cyc in stem:
            return "WLTC_HWY" if cyc == "WLTC" else cyc
    return "WLTC_HWY"


def _to_kelvin(arr: np.ndarray, unit: str = "") -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).ravel()
    u = unit.strip().lower()
    if u in ("degc", "c", "celsius") or (u == "" and np.nanmean(arr) < 150.0):
        return arr + 273.15
    return arr


def _to_pascal(arr: np.ndarray, unit: str = "") -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).ravel()
    u = unit.strip().lower()
    if u == "mpa":
        return arr * 1e6
    if u == "bar":
        return arr * 1e5
    if u == "kpa":
        return arr * 1e3
    if u == "pa":
        return arr
    if np.nanmax(arr) < 8.0:
        return arr * 1e6
    if np.nanmax(arr) < 100.0:
        return arr * 1e5
    return arr


def load_simlog_export(path: Path, repo_root: Path, cycle: Optional[str] = None) -> ScenarioData:
    """Load flattened simlog_*_export.mat (exact parity with validate_com_fig2.m)."""
    m = sio.loadmat(path, squeeze_me=True)
    names = [str(n) for n in m["names"]]
    units = [str(u) for u in m["units"]] if "units" in m else [""] * len(names)
    data = np.asarray(m["data"], dtype=np.float64)
    t = np.asarray(m["time"], dtype=np.float64).ravel()

    name_to_idx = {n: i for i, n in enumerate(names)}

    def get_sig(sig: str) -> Tuple[np.ndarray, str]:
        key = sig if sig.startswith("simlog.") else f"simlog.{sig}"
        if key not in name_to_idx:
            raise KeyError(f"Signal '{key}' missing in {path}")
        idx = name_to_idx[key]
        return data[:, idx].copy(), units[idx]

    mot_t, u_mot = get_sig("Motor.Motor.M.T")
    inv_t, u_inv = get_sig("Inverter.Inverter.M.T")
    dcdc_t, u_dcdc = get_sig("DCDC.DC_DC_Converter.M.T")
    cab_air_t, u_cab = get_sig("Cabin.Cabin_Air_Volume.T_I")

    pack_t = np.zeros_like(t)
    soc = np.zeros_like(t)
    for k in range(1, 5):
        bt, u_bt = get_sig(f"Battery.Pack_{k}.Thermal_Model.Battery_Thermal_Mass.T")
        sc, _ = get_sig(f"Battery.Pack_{k}.Lithium_Cell_1RC.Em_table.SOC")
        pack_t += _to_kelvin(bt, u_bt) / 4.0
        soc += sc / 4.0

    # Thermal-mass weighted cabin interior temperature matching validate_com_fig2.m L149-153
    wall_c = np.array([36.4 * 840.0, 75.0 * 500.0, 30.0 * 500.0], dtype=np.float64)
    t_glass, u_g = get_sig("Cabin.Cabin_Heat_Transfer.Glass_Thermal_Mass.M.T")
    t_doors, u_d = get_sig("Cabin.Cabin_Heat_Transfer.Doors_Thermal_Mass.M.T")
    t_roof, u_r = get_sig("Cabin.Cabin_Heat_Transfer.Roof_Thermal_Mass.M.T")
    walls = np.column_stack([_to_kelvin(t_glass, u_g), _to_kelvin(t_doors, u_d), _to_kelvin(t_roof, u_r)])
    cab_int_t = (walls @ wall_c) / wall_c.sum()

    p_in_raw, u_pin = get_sig("Compressor.Compressor.A.p")
    p_out_raw, u_pout = get_sig("Compressor.Compressor.B.p")

    X = np.column_stack([
        _to_kelvin(mot_t, u_mot),
        _to_kelvin(inv_t, u_inv),
        _to_kelvin(dcdc_t, u_dcdc),
        soc,
        pack_t,
        _to_pascal(p_in_raw, u_pin),
        _to_pascal(p_out_raw, u_pout),
        cab_int_t,
        _to_kelvin(cab_air_t, u_cab),
    ])

    comp_w, _ = get_sig("Compressor.Compressor.omega")
    blow_m, _ = get_sig("Blower.Blower.mdot_A")
    mp_w, _ = get_sig("Motor_Pump.Motor_Pump.omega")
    bp_w, _ = get_sig("Battery_Pump.Battery_Pump.omega")
    htr_q, _ = get_sig("Heater.Controlled_Heat_Flow_Rate_Source.S")
    fan_w, _ = get_sig("Radiator.Fan.omega")

    U = np.column_stack([comp_w, blow_m, mp_w, bp_w, htr_q, fan_w])

    mot_q, _ = get_sig("Motor.Controlled_Heat_Flow_Rate_Source.S")
    inv_q, _ = get_sig("Inverter.Controlled_Heat_Flow_Rate_Source.S")
    dcdc_q, _ = get_sig("DCDC.Controlled_Heat_Flow_Rate_Source.S")
    bat_i, _ = get_sig("Battery.Current_Demand.i")
    env_t_arr, u_env = get_sig("Cabin.Environment_Air.T")
    env_t_k = _to_kelvin(env_t_arr, u_env)
    tamb_c = float(np.median(env_t_k) - 273.15)

    clnt_bat_raw, u_cbat = get_sig("Heater.Heater_Coolant_Volume.T_I")
    clnt_bat_k = _to_kelvin(clnt_bat_raw, u_cbat)

    if "simlog.Inverter.Coolant_Jacket.B.T" in name_to_idx:
        clnt_inv_raw, u_cinv = get_sig("Inverter.Coolant_Jacket.B.T")
    else:
        clnt_inv_raw, u_cinv = get_sig("Inverter.Coolant_Jacket.A.T")
    clnt_inv_k = _to_kelvin(clnt_inv_raw, u_cinv)

    cyc = cycle or _infer_cycle_from_name(path)
    speed_kmh = _load_baseline_speed(repo_root, t)

    return ScenarioData(
        name=f"{cyc}_T{int(round(tamb_c))}",
        cycle=cyc,
        tamb_c=tamb_c,
        source_path=path,
        source_format="simlog_export",
        t=t,
        X=X,
        U=U,
        motQ=mot_q,
        invQ=inv_q,
        dcdcQ=dcdc_q,
        batI=bat_i,
        speed_kmh=speed_kmh,
        clntInvTout_K=clnt_inv_k,
        clntBatTin_K=clnt_bat_k,
        batI_avg=causal_moving_avg(bat_i, window=120),
    )


def load_training_data_mat(path: Path) -> ScenarioData:
    """Load Abdiraham's training_data_rule_*.mat containing `trainingData` struct."""
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    td = m["trainingData"]

    t = np.asarray(td.t_s, dtype=np.float64).ravel()
    tamb_c = float(np.asarray(td.T_amb_C, dtype=np.float64).ravel()[0])
    cyc = str(td.cycle).strip() if hasattr(td, "cycle") else _infer_cycle_from_name(path)

    # Prefer thermal-mass weighted cabIntTw_C when available (NEDC/UDDS/US06), else cabIntT_C
    if hasattr(td, "cabIntTw_C"):
        cab_int_k = np.asarray(td.cabIntTw_C, dtype=np.float64).ravel() + 273.15
    else:
        cab_int_k = np.asarray(td.cabIntT_C, dtype=np.float64).ravel() + 273.15

    X = np.column_stack([
        np.asarray(td.motT_C, dtype=np.float64).ravel() + 273.15,
        np.asarray(td.invT_C, dtype=np.float64).ravel() + 273.15,
        np.asarray(td.dcdcT_C, dtype=np.float64).ravel() + 273.15,
        np.asarray(td.batSoc, dtype=np.float64).ravel(),
        np.asarray(td.batT_C, dtype=np.float64).ravel() + 273.15,
        np.asarray(td.compPin_MPa, dtype=np.float64).ravel() * 1e6,
        np.asarray(td.compPout_MPa, dtype=np.float64).ravel() * 1e6,
        cab_int_k,
        np.asarray(td.cabAirT_C, dtype=np.float64).ravel() + 273.15,
    ])

    U = np.column_stack([
        np.asarray(td.compNrpm, dtype=np.float64).ravel(),
        np.asarray(td.cabAirMfIn_kgs, dtype=np.float64).ravel(),
        np.asarray(td.motPumpNrpm, dtype=np.float64).ravel(),
        np.asarray(td.batPumpNrpm, dtype=np.float64).ravel(),
        np.asarray(td.heaterQ_W, dtype=np.float64).ravel(),
        np.asarray(td.radFanrpm, dtype=np.float64).ravel(),
    ])

    bat_i = np.asarray(td.batI_A, dtype=np.float64).ravel()
    return ScenarioData(
        name=f"{cyc}_T{int(round(tamb_c))}",
        cycle=cyc,
        tamb_c=tamb_c,
        source_path=path,
        source_format="training_data",
        t=t,
        X=X,
        U=U,
        motQ=np.asarray(td.motQgen_W, dtype=np.float64).ravel(),
        invQ=np.asarray(td.invQgen_W, dtype=np.float64).ravel(),
        dcdcQ=np.asarray(td.dcdcQgen_W, dtype=np.float64).ravel(),
        batI=bat_i,
        speed_kmh=np.asarray(td.vehSpeed_kmh, dtype=np.float64).ravel(),
        clntInvTout_K=np.asarray(td.clntInvTin_C, dtype=np.float64).ravel() + 273.15,
        clntBatTin_K=np.asarray(td.heaterClntTout_C, dtype=np.float64).ravel() + 273.15,
        batI_avg=causal_moving_avg(bat_i, window=120),
    )


def discover_scenarios(repo_root: Path) -> List[ScenarioData]:
    """Discover all simlog_*_export.mat and Results/**/training_data_*.mat runs."""
    candidates: List[Path] = []
    candidates.extend(sorted(repo_root.glob("simlog_*_export.mat")))
    exported_dir = repo_root / "Results" / "exported"
    if exported_dir.is_dir():
        candidates.extend(sorted(exported_dir.rglob("*_export.mat")))
    results_dir = repo_root / "Results"
    if results_dir.is_dir():
        candidates.extend(sorted(results_dir.rglob("training_data_*.mat")))

    by_key: Dict[Tuple[str, int], ScenarioData] = {}
    for p in candidates:
        if p.name.endswith("_export.mat"):
            sc = load_simlog_export(p, repo_root)
        else:
            sc = load_training_data_mat(p)
        key = (sc.cycle, int(round(sc.tamb_c)))
        # Prefer simlog_export over training_data if both exist for the exact same (cycle, T_amb)
        if key not in by_key or sc.source_format == "simlog_export":
            by_key[key] = sc

    scenarios = sorted(by_key.values(), key=lambda s: (s.cycle, -s.tamb_c))
    return scenarios
