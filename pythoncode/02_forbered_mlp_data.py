"""Kontrollera verkliga MAT-filer och förbered KÄNDA tillstånd på 1 s-grid.

Resultatet märks PARTIAL och får inte användas som ett komplett COM-träningsset.
Saknade tillstånd lämnas som NaN med en separat mask, aldrig som påhittade nollor.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
RUN = "run_2026_11_11_10_22_21"
STATE_NAMES = ["motT_K", "invT_K", "dcdcT_K", "batSoc", "batT_K",
               "compPin_Pa", "compPout_Pa", "cabIntT_K", "cabAirT_K"]


def interpolate_temperature(signal, grid):
    if signal["unit"] != "degC":
        raise ValueError("Förväntade temperatur i degC")
    t = np.asarray(signal["t_s"], float).reshape(-1)
    y = np.asarray(signal["y"], float).reshape(-1)
    if t.size != y.size or t.size < 2 or not (np.isfinite(t).all() and np.isfinite(y).all()):
        raise ValueError("Ogiltig temperaturserie")
    if (np.diff(t) < 0).any():
        raise ValueError("Tiden går bakåt")
    # Lösaren ger ibland nästan samma tid vid en händelse. Behåll sista värdet
    # i grupper där efterföljande tidssteg är <= 1 ns. Ingen extrapolation tillåts.
    keep = np.r_[np.diff(t) > 1e-9, True]
    t, y = t[keep], y[keep]
    if grid[0] < t[0]-1e-9 or grid[-1] > t[-1]+1e-9:
        raise ValueError("Signalens tid täcker inte gridet")
    return np.interp(grid, t, y) + 273.15  # COM använder Kelvin.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=Path,
                        default=ROOT / "Data" / "NN_exports" / RUN)
    parser.add_argument("--output-folder", type=Path,
                        default=ROOT / "Data" / "NN_training" / "prepared_partial")
    args = parser.parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)
    report = {
        "training_ready": False,
        "sample_time_s": 1.0,
        "horizon_from_init_nmpc": 30,
        "architecture": [11, 32, 32, 11],
        "state_order": STATE_NAMES,
        "missing_states": ["batSoc", "compPin_Pa", "compPout_Pa", "cabIntT_K"],
        "missing_inputs": ["Ib_A", "compNrpm", "cabAirMfIn", "motPumpNrpm",
                           "batPumpNrpm", "heaterPwr", "radFanrpm"],
        "further_requirements": [
            "63 ordered fixed est/ctrl parameters or a verified construction of them",
            "reviewed gamma bounds and nominal values; interpolation baseline",
            "verify the battery-average state against the actual COM measurement mapping",
        ],
        "runs": [],
    }
    for tamb in (-5, -7, -10):
        file = args.input_folder / f"rule_T{abs(tamb)}_numeric.mat"
        run = loadmat(file, simplify_cells=True)["run_data"]
        if float(run["T_amb_C"]) != tamb:
            raise ValueError(f"Fel temperaturmetadata i {file}")
        signals = run["signals"]
        # Ts=1 s kommer från init_nmpc.m. Detta första steg gäller temperaturer;
        # diskreta styrkommandon måste senare hanteras med lämplig hållfunktion.
        grid = np.arange(0.0, 3600.0 + 1.0, 1.0)
        x = np.full((grid.size, 9), np.nan)
        for col, name in ((0, "T_motor_C"), (1, "T_inverter_C"),
                          (2, "T_dcdc_C"), (8, "T_cabin_C")):
            x[:, col] = interpolate_temperature(signals[name], grid)
        # Medelvärdet följer er tidigare plottningskod som första kandidat.
        # Alla fyra pack interpoleras FÖRST till samma tidsgrid.
        packs = [interpolate_temperature(signals[f"T_bat_pack{i}_C"], grid)
                 for i in range(1, 5)]
        x[:, 4] = np.mean(packs, axis=0)
        available = np.isfinite(x).all(axis=0)
        output = args.output_folder / f"PARTIAL_T{abs(tamb)}.npz"
        np.savez_compressed(output, t_s=grid, x_partial=x, observed_states=available,
                            state_names=np.array(STATE_NAMES), T_amb_C=tamb,
                            training_ready=False)
        all_finite = all(np.isfinite(s["y"]).all() and np.isfinite(s["t_s"]).all()
                         for s in signals.values())
        same_time = all(np.array_equal(s["t_s"], signals["T_cabin_C"]["t_s"])
                        for s in signals.values())
        details = {"file": file.name, "T_amb_C": tamb, "signals": len(signals),
                   "native_samples": int(np.asarray(signals["T_cabin_C"]["y"]).size),
                   "all_finite": bool(all_finite), "same_time_within_run": same_time,
                   "grid_samples": grid.size, "available_state_count": int(available.sum()),
                   "output": output.name}
        report["runs"].append(details)
        print(f"{file.name}: {details['native_samples']} original samples, "
              f"{grid.size} at 1 s, {available.sum()}/9 states available")
    report_file = args.output_folder / "data_readiness.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nPARTIAL preparation complete. Vehicle training is not ready yet.")
    print("Missing states:", ", ".join(report["missing_states"]))
    print("Report:", report_file)


if __name__ == "__main__":
    main()
