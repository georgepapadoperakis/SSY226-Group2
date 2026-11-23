"""
Differentiable lookup-table version of fluid_props_coolprop.estimate_params, for training free runs.

fluid_props_coolprop calls CoolProp for every second and every run (slow, and gradients cannot flow through it).
With the superheat fixed (SUPERHEAT_K) every property depends on only one of
    p_in (low side), p_out (high side), cabin-air temperature, or (p_in, p_out) for the isentropic outlet enthalpy,
so they are tabulated once on fine grids and linearly interpolated in torch. The paper suggests the same
("pre-computing the properties ... in lookup tables", Sec. VII-D).

    tab = PropTable.build(superheat_K)          # ~1 min of CoolProp calls, then cached in data_cache/fluid_property_table.pt
    est = tab(x, heat_k, envT_K)                # x [B, 9] COM state -> dict of [B] tensors (EST_NAMES)
"""
import math
from pathlib import Path
import numpy as np
import torch
import CoolProp.CoolProp as CP

DT = torch.float64
REF, AIR, P_ATM = "R134a", "Air", 101325.0
P_LO, P_HI = 0.2e5, 40.0e5          # table range for both pressures (CoolProp saturation limit ~40.6 bar)


def _ps(out, n1, v1, n2, v2, fluid):
    return CP.PropsSI(out, n1, float(v1), n2, float(v2), fluid)


def _interp1(xq, xg, yg):
    """Linear interpolation, clamped at the grid ends; differentiable in xq. xg [G] increasing, yg [G, F]."""
    xq = xq.clamp(xg[0], xg[-1])
    i = torch.searchsorted(xg, xq.detach()).clamp(1, len(xg) - 1)
    x0, x1 = xg[i - 1], xg[i]
    w = ((xq - x0) / (x1 - x0)).unsqueeze(-1)
    return yg[i - 1] * (1 - w) + yg[i] * w


def _interp2(aq, bq, ag, bg, zg):
    """Bilinear interpolation on a [Ga, Gb] grid, clamped; differentiable in aq, bq."""
    aq, bq = aq.clamp(ag[0], ag[-1]), bq.clamp(bg[0], bg[-1])
    i = torch.searchsorted(ag, aq.detach()).clamp(1, len(ag) - 1)
    j = torch.searchsorted(bg, bq.detach()).clamp(1, len(bg) - 1)
    wa = (aq - ag[i - 1]) / (ag[i] - ag[i - 1]); wb = (bq - bg[j - 1]) / (bg[j] - bg[j - 1])
    return (zg[i - 1, j - 1] * (1 - wa) * (1 - wb) + zg[i, j - 1] * wa * (1 - wb)
            + zg[i - 1, j] * (1 - wa) * wb + zg[i, j] * wa * wb)


LOW = ["refTsatLowPK", "compTinK", "compHin", "compSin", "condEvapRefRho", "condEvapRefK", "condEvapRefMu",
       "condEvapRefCp", "refHsatVapLowP", "refTsatLiqLowPK", "refCpLiqLowP",
       "dTesat_dp1", "d_rhoh_l_dp1", "d_rhoh_g_dp1"]
HIGH = ["refTsatHighPK", "refCpVapHighP", "innerCondRefRho", "innerCondRefK", "innerCondRefMu", "innerCondRefCp",
        "innerCondRefHliq", "innerCondRefHvap", "refRhoVapHighP", "refRhoLiqHighP", "refTsatLiqHighPK",
        "refHsatLiqHighP", "refCpLiqHighP", "dTcsat_dp2", "d_rhoh_l_dp2", "d_rhoh_g_dp2"]
AIRF = ["cabAirCp", "cabAirK", "cabAirMu", "cabAirRho"]


def _low(p, sh):
    e = {"refTsatLowPK": _ps("T", "P", p, "Q", 1, REF)}
    e["compTinK"] = e["refTsatLowPK"] + abs(sh)
    h1 = _ps("H", "P", p, "Q", 1, REF) if abs(sh) < 0.5 else _ps("H", "P", p, "T", e["compTinK"], REF)
    e["compHin"], e["compSin"] = h1, _ps("S", "P", p, "H", h1, REF)
    for q, n in zip("DLVC", ("Rho", "K", "Mu", "Cp")):
        e["condEvapRef" + n] = _ps(q, "P", p, "Q", 0, REF)
    e["refHsatVapLowP"] = _ps("H", "P", p, "Q", 1, REF)
    e["refTsatLiqLowPK"] = _ps("T", "P", p, "Q", 0, REF)
    e["refCpLiqLowP"] = _ps("C", "P", p, "Q", 0, REF)
    d = lambda fn: (fn(p + 100.0) - fn(p - 100.0)) / 200.0
    e["dTesat_dp1"] = d(lambda q: _ps("T", "P", q, "Q", 1, REF))
    e["d_rhoh_l_dp1"] = d(lambda q: _ps("D", "P", q, "Q", 0, REF) * _ps("H", "P", q, "Q", 0, REF))
    e["d_rhoh_g_dp1"] = d(lambda q: _ps("D", "P", q, "Q", 1, REF) * _ps("H", "P", q, "Q", 1, REF))
    return [e[n] for n in LOW]


def _high(p):
    e = {}
    try:
        e["refTsatHighPK"], e["refCpVapHighP"] = _ps("T", "P", p, "Q", 1, REF), _ps("C", "P", p, "Q", 1, REF)
    except ValueError:
        e["refTsatHighPK"], e["refCpVapHighP"] = _ps("T", "P", p + 100, "Q", 1, REF), _ps("C", "P", p + 100, "Q", 1, REF)
    for q, n in zip("DLVC", ("Rho", "K", "Mu", "Cp")):
        e["innerCondRef" + n] = _ps(q, "P", p, "Q", 1, REF)
    e["innerCondRefHliq"], e["innerCondRefHvap"] = _ps("H", "P", p, "Q", 0, REF), _ps("H", "P", p, "Q", 1, REF)
    e["refRhoVapHighP"], e["refRhoLiqHighP"] = _ps("D", "P", p, "Q", 1, REF), _ps("D", "P", p, "Q", 0, REF)
    e["refTsatLiqHighPK"] = _ps("T", "P", p, "Q", 0, REF)
    e["refHsatLiqHighP"], e["refCpLiqHighP"] = _ps("H", "P", p, "Q", 0, REF), _ps("C", "P", p, "Q", 0, REF)
    d = lambda fn: (fn(p + 100.0) - fn(p - 100.0)) / 200.0
    e["dTcsat_dp2"] = d(lambda q: _ps("T", "P", q, "Q", 1, REF))
    e["d_rhoh_l_dp2"] = d(lambda q: _ps("D", "P", q, "Q", 0, REF) * _ps("H", "P", q, "Q", 0, REF))
    e["d_rhoh_g_dp2"] = d(lambda q: _ps("D", "P", q, "Q", 1, REF) * _ps("H", "P", q, "Q", 1, REF))
    return [e[n] for n in HIGH]


def _fill_nan(a):
    """Replace NaN entries (CoolProp failures at a few grid points) by the nearest valid value along axis 0."""
    a = np.array(a, float)
    for f in range(a.shape[1]):
        col, ok = a[:, f], np.isfinite(a[:, f])
        if not ok.all():
            idx = np.arange(len(col))
            col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
    return a


class PropTable:
    def __init__(self, d):
        self.d = d
        self.const = d["const"]

    @classmethod
    def build(cls, superheat_K, path=Path(__file__).absolute().parent / "data_cache" / "fluid_property_table.pt", n_p=500, n_2d=90):
        path = Path(path)
        if path.exists():
            d = torch.load(path, weights_only=False)
            if abs(d["superheat_K"] - superheat_K) < 1e-9:
                return cls(d)
        pg = np.geomspace(P_LO, P_HI, n_p)
        low = _fill_nan([_safe(lambda p=p: _low(p, superheat_K), len(LOW)) for p in pg])
        high = _fill_nan([_safe(lambda p=p: _high(p), len(HIGH)) for p in pg])
        tg = np.linspace(220.0, 340.0, 241)
        air = np.array([[_ps(q, "T", T, "P", P_ATM, AIR) for q in "CLVD"] for T in tg])
        # isentropic outlet enthalpy h2s(p_out, s1(p_in))
        p2 = np.geomspace(P_LO, P_HI, n_2d)
        s1 = np.array([_safe(lambda p=p: [_low(p, superheat_K)[LOW.index("compSin")]], 1)[0] for p in p2])
        h2s = np.full((n_2d, n_2d), np.nan)
        for i, s in enumerate(s1):
            for j, po in enumerate(p2):
                try:
                    h2s[i, j] = _ps("H", "P", po, "S", s, REF)
                except ValueError:
                    try:
                        h2s[i, j] = _ps("H", "P", po + 100, "S", s, REF)
                    except ValueError:
                        pass
        h2s = _fill_nan(h2s)
        envAirT = 263.15
        const = {"compVnominalIn": 1.0 / _ps("D", "P", 3e5, "T", 11 + 273.15, REF),
                 "condEvapAirRho": _ps("D", "T", envAirT, "P", P_ATM, AIR),
                 "condEvapAirK": _ps("L", "T", envAirT, "P", P_ATM, AIR),
                 "condEvapAirMu": _ps("V", "T", envAirT, "P", P_ATM, AIR),
                 "condEvapAirCp": _ps("C", "T", envAirT, "P", P_ATM, AIR),
                 "clntRho": 1113.0, "clntK": 0.254, "clntMu": 16.1e-3, "clntCp": 2430.0,
                 "evapVoidFrac": 0.8, "condVoidFrac": 0.5, "evapVol_m3": 0.001, "condVol_m3": 0.001,
                 "evapWallThermalMass_J_K": 5000.0, "condWallThermalMass_J_K": 5000.0}
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=DT)
        d = dict(superheat_K=superheat_K, pg=t(np.log(pg)), low=t(low), high=t(high), tg=t(tg), air=t(air),
                 p2=t(np.log(p2)), h2s=t(h2s), const=const)
        torch.save(d, path)
        return cls(d)

    def __call__(self, x, heat=None, envT_K=None):
        """x [B, 9] COM state (K, Pa). heat: dict of [B] tensors (motQmeas, invQmeas, dcdcQmeas, batImeas, speed)."""
        d = self.d
        lp_in = torch.log(x[:, 5].clamp(P_LO, P_HI)); lp_out = torch.log(x[:, 6].clamp(P_LO, P_HI))
        L = _interp1(lp_in, d["pg"], d["low"]); Hh = _interp1(lp_out, d["pg"], d["high"])
        A = _interp1(x[:, 8], d["tg"], d["air"])
        e = {n: L[:, i] for i, n in enumerate(LOW)}
        e.update({n: Hh[:, i] for i, n in enumerate(HIGH)})
        e.update({n: A[:, i] for i, n in enumerate(AIRF)})
        e["compHoutS"] = _interp2(lp_in, lp_out, d["p2"], d["p2"], d["h2s"])
        for pre in ("evapRef", "chilRef"):
            for n in ("Rho", "K", "Mu", "Cp"):
                e[pre + n] = e["condEvapRef" + n]
        B = x.shape[0]
        for n, v in self.const.items():
            e[n] = torch.full((B,), v, dtype=DT)
        if envT_K is not None:
            e["envT"] = envT_K
        if heat is not None:
            e.update(heat)
        return e


def _safe(fn, n):
    try:
        return fn()
    except ValueError:
        return [np.nan] * n
