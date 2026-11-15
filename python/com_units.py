"""
Convert exported 1-Hz training data (training_data_<ctrl>_T<|T|>.mat) to the units the
COM (Source Files/dynamics_mexfreeNew.m) uses, and check that the result is plausible.

COM units, taken from MPCController_stepNew.m (lines 47-55) and dynamics_mexfreeNew.m:
  states  x : temperatures in K, pressures in Pa, SOC in [-]
  inputs  u : compNrpm [rpm], cabAirMfIn [kg/s], motPumpNrpm/batPumpNrpm [rpm],
              heaterPwr [W], radFanrpm [rpm]
  params    : envT [K], speed [km/h] (COM divides by 3.6 itself), batImeas [A],
              motQmeas/invQmeas/dcdcQmeas [W]
"""
import numpy as np
import scipy.io as sio

C2K, MPA2PA = 273.15, 1e6

# COM state order (x(1)..x(9)) -> (export field, offset, scale): com = (export + offset) * scale
STATES = [
    ("motT",       "motT_C",       C2K, 1.0),
    ("invT",       "invT_C",       C2K, 1.0),
    ("dcdcT",      "dcdcT_C",      C2K, 1.0),
    ("batSoc",     "batSoc",       0.0, 1.0),
    ("batT",       "batT_C",       C2K, 1.0),
    ("compPinPa",  "compPin_MPa",  0.0, MPA2PA),
    ("compPoutPa", "compPout_MPa", 0.0, MPA2PA),
    ("cabIntT",    "cabIntT_C",    C2K, 1.0),   # NB: export is a plain average of the walls; COM is mass-weighted
    ("cabAirT",    "cabAirT_C",    C2K, 1.0),
]
INPUTS = [  # already in COM units
    ("compNrpm",    "compNrpm"),
    ("cabAirMfIn",  "cabAirMfIn_kgs"),
    ("motPumpNrpm", "motPumpNrpm"),
    ("batPumpNrpm", "batPumpNrpm"),
    ("heaterPwr",   "heaterQ_W"),
    ("radFanrpm",   "radFanrpm"),
]
PARAMS = [  # disturbances / measured signals
    ("speed",     "vehSpeed_kmh", 0.0, 1.0),   # km/h, NOT m/s
    ("batImeas",  "batI_A",       0.0, 1.0),
    ("motQmeas",  "motQgen_W",    0.0, 1.0),
    ("invQmeas",  "invQgen_W",    0.0, 1.0),
    ("dcdcQmeas", "dcdcQgen_W",   0.0, 1.0),
]

# Plausible physical ranges in COM units. A unit mistake (degC vs K, MPa vs Pa, m/s vs km/h)
# lands far outside these, so the check catches it.
RANGES = {
    "motT": (220, 420), "invT": (220, 420), "dcdcT": (220, 420), "batT": (220, 350),
    "cabIntT": (220, 350), "cabAirT": (220, 350), "batSoc": (0, 1),
    "compPinPa": (2e4, 1e6), "compPoutPa": (2e4, 6e6),
    "compNrpm": (0, 1e4), "cabAirMfIn": (0, 1), "motPumpNrpm": (0, 1e4), "batPumpNrpm": (0, 1e4),
    "heaterPwr": (0, 4000.5), "radFanrpm": (0, 5000),
    "speed": (0, 250), "batImeas": (-500, 500), "motQmeas": (0, 2e4), "invQmeas": (0, 5e3), "dcdcQmeas": (0, 5e3),
    "envT": (220, 320),
}


def load_export(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    td = m["trainingData"]
    return {k: np.asarray(getattr(td, k)) for k in td._fieldnames}


def to_com(d):
    """Return dict with x [N,9], u [N,6], p {name: [N]}, envT [K], t [s] in COM units."""
    x = np.stack([(np.asarray(d[f], float) + off) * sc for _, f, off, sc in STATES], axis=1)
    u = np.stack([np.asarray(d[f], float) for _, f in INPUTS], axis=1)
    p = {n: (np.asarray(d[f], float) + off) * sc for n, f, off, sc in PARAMS}
    envT = float(d["T_amb_C"]) + C2K
    return {"t": np.asarray(d["t_s"], float), "x": x, "u": u, "p": p, "envT": envT,
            "x_names": [s[0] for s in STATES], "u_names": [i[0] for i in INPUTS]}


def check(com, verbose=True):
    """Raise if any signal is outside its plausible COM-unit range or not finite."""
    problems = []
    cols = [(n, com["x"][:, i]) for i, n in enumerate(com["x_names"])] + \
           [(n, com["u"][:, i]) for i, n in enumerate(com["u_names"])] + \
           list(com["p"].items()) + [("envT", np.array([com["envT"]]))]
    for n, v in cols:
        lo, hi = RANGES[n]
        bad = ~np.isfinite(v) | (v < lo) | (v > hi)
        if verbose:
            print(f"  {n:12s} {np.nanmin(v):12.4g} .. {np.nanmax(v):12.4g}   [{lo:g}, {hi:g}]  {'OK' if not bad.any() else 'OUT OF RANGE'}")
        if bad.any():
            problems.append(f"{n}: {bad.sum()} samples outside [{lo}, {hi}]")
    if problems:
        raise ValueError("Unit/range check failed:\n  " + "\n  ".join(problems))
    return True


if __name__ == "__main__":
    import glob, os, sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "Results")
    for f in sorted(glob.glob(os.path.join(root, "**", "training_data_*.mat"), recursive=True)):
        print(f"\n{os.path.basename(f)}")
        check(to_com(load_export(f)))
    print("\nAll files passed.")
