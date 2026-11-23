"""
Python port of Source Files/estimateParameters.m: the per-second refrigerant / air / coolant properties
("estimatedParams") that the COM needs, computed with CoolProp from measured pressures.

    est = estimate_params(compPinPa, compPoutPa, cabAirT_K, superheatK, envT_K, motQ, invQ, dcdcQ, batI, speed)

All inputs are 1-D arrays over time (length N). Returns a dict of length-N numpy arrays, with the same field
names as in MATLAB (and as com_torch.EST_NAMES). Needs `pip install CoolProp`.
"""
import numpy as np
import CoolProp.CoolProp as CP

REF, AIR, P_ATM = "R134a", "Air", 101325.0


def _ps(out, n1, v1, n2, v2, fluid):
    return CP.PropsSI(out, n1, float(v1), n2, float(v2), fluid)


def _one(pin, pout, cabT, sh):
    """estimateParameters.m for one time step (scalar inputs)."""
    e = {}
    e["superheatK"] = sh
    e["compVnominalIn"] = 1.0 / _ps("D", "P", 3e5, "T", 11 + 273.15, REF)
    e["refTsatLowPK"] = _ps("T", "P", pin, "Q", 1, REF)
    e["compTinK"] = e["refTsatLowPK"] + abs(sh)
    if abs(e["compTinK"] - e["refTsatLowPK"]) < 0.5:
        h1 = _ps("H", "P", pin, "Q", 1, REF)
    else:
        h1 = _ps("H", "P", pin, "T", e["compTinK"], REF)
    s1 = _ps("S", "P", pin, "H", h1, REF)
    try:
        h2s = _ps("H", "P", pout, "S", s1, REF)
        e["refTsatHighPK"] = _ps("T", "P", pout, "Q", 1, REF)
        e["refCpVapHighP"] = _ps("C", "P", pout, "Q", 1, REF)
    except ValueError:                                     # MATLAB: catch -> retry at p + 100 Pa
        h2s = _ps("H", "P", pout + 100, "S", s1, REF)
        e["refTsatHighPK"] = _ps("T", "P", pout + 100, "Q", 1, REF)
        e["refCpVapHighP"] = _ps("C", "P", pout + 100, "Q", 1, REF)
    e["compHin"], e["compHoutS"], e["compSin"] = h1, h2s, s1

    # inner condenser, air side at cabin-air temperature
    e["cabAirCp"] = _ps("C", "T", cabT, "P", P_ATM, AIR)
    e["cabAirK"] = _ps("L", "T", cabT, "P", P_ATM, AIR)
    e["cabAirMu"] = _ps("V", "T", cabT, "P", P_ATM, AIR)
    e["cabAirRho"] = _ps("D", "T", cabT, "P", P_ATM, AIR)
    e["innerCondRefRho"] = _ps("D", "P", pout, "Q", 1, REF)
    e["innerCondRefK"] = _ps("L", "P", pout, "Q", 1, REF)
    e["innerCondRefMu"] = _ps("V", "P", pout, "Q", 1, REF)
    e["innerCondRefCp"] = _ps("C", "P", pout, "Q", 1, REF)
    e["innerCondRefHliq"] = _ps("H", "P", pout, "Q", 0, REF)
    e["innerCondRefHvap"] = _ps("H", "P", pout, "Q", 1, REF)
    e["refRhoVapHighP"] = _ps("D", "P", pout, "Q", 1, REF)
    e["refRhoLiqHighP"] = _ps("D", "P", pout, "Q", 0, REF)

    # evaporator side (saturated liquid at low pressure), shared by condEvap / evap / chiller
    rho, k, mu, cp = (_ps(q, "P", pin, "Q", 0, REF) for q in ("D", "L", "V", "C"))
    for pre in ("condEvapRef", "evapRef", "chilRef"):
        e[pre + "Rho"], e[pre + "K"], e[pre + "Mu"], e[pre + "Cp"] = rho, k, mu, cp

    # front heat exchanger air: fixed at -10 °C in the MATLAB code
    envAirT = 263.15
    e["condEvapAirRho"] = _ps("D", "T", envAirT, "P", P_ATM, AIR)
    e["condEvapAirK"] = _ps("L", "T", envAirT, "P", P_ATM, AIR)
    e["condEvapAirMu"] = _ps("V", "T", envAirT, "P", P_ATM, AIR)
    e["condEvapAirCp"] = _ps("C", "T", envAirT, "P", P_ATM, AIR)

    e["refTsatLiqHighPK"] = _ps("T", "P", pout, "Q", 0, REF)
    e["refHsatLiqHighP"] = _ps("H", "P", pout, "Q", 0, REF)
    e["refCpLiqHighP"] = _ps("C", "P", pout, "Q", 0, REF)
    e["refHsatVapLowP"] = _ps("H", "P", pin, "Q", 1, REF)
    e["refTsatLiqLowPK"] = _ps("T", "P", pin, "Q", 0, REF)
    e["refCpLiqLowP"] = _ps("C", "P", pin, "Q", 0, REF)

    # finite-difference derivatives for the pressure dynamics
    dp = 100.0
    def d_dp(p, fn):
        return (fn(p + dp) - fn(p - dp)) / (2 * dp)
    e["dTesat_dp1"] = d_dp(pin, lambda p: _ps("T", "P", p, "Q", 1, REF))
    e["d_rhoh_l_dp1"] = d_dp(pin, lambda p: _ps("D", "P", p, "Q", 0, REF) * _ps("H", "P", p, "Q", 0, REF))
    e["d_rhoh_g_dp1"] = d_dp(pin, lambda p: _ps("D", "P", p, "Q", 1, REF) * _ps("H", "P", p, "Q", 1, REF))
    e["dTcsat_dp2"] = d_dp(pout, lambda p: _ps("T", "P", p, "Q", 1, REF))
    e["d_rhoh_l_dp2"] = d_dp(pout, lambda p: _ps("D", "P", p, "Q", 0, REF) * _ps("H", "P", p, "Q", 0, REF))
    e["d_rhoh_g_dp2"] = d_dp(pout, lambda p: _ps("D", "P", p, "Q", 1, REF) * _ps("H", "P", p, "Q", 1, REF))
    return e


def estimate_params(compPinPa, compPoutPa, cabAirT_K, superheatK, envT_K,
                    motQ, invQ, dcdcQ, batI, speed=None):
    N = len(compPinPa)
    # CoolProp's R134a model is valid below the critical pressure (40.59 bar) for saturation calls;
    # clip so a short start-up spike above it does not crash (the MATLAB code would error there too).
    pout = np.clip(np.asarray(compPoutPa, float), 1e4, 4.0e6)
    pin = np.clip(np.asarray(compPinPa, float), 1e4, 4.0e6)
    rows = [_one(pin[k], pout[k], float(cabAirT_K[k]), float(superheatK[k])) for k in range(N)]
    est = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    est.update(clntRho=np.full(N, 1113.0), clntK=np.full(N, 0.254), clntMu=np.full(N, 16.1e-3),
               clntCp=np.full(N, 2430.0),
               envT=np.full(N, float(envT_K)) if np.ndim(envT_K) == 0 else np.asarray(envT_K, float),
               motQmeas=np.asarray(motQ, float), invQmeas=np.asarray(invQ, float),
               dcdcQmeas=np.asarray(dcdcQ, float), batImeas=np.asarray(batI, float),
               speed=np.zeros(N) if speed is None else np.asarray(speed, float),
               evapVoidFrac=np.full(N, 0.8), condVoidFrac=np.full(N, 0.5),
               evapVol_m3=np.full(N, 0.001), condVol_m3=np.full(N, 0.001),
               evapWallThermalMass_J_K=np.full(N, 5000.0), condWallThermalMass_J_K=np.full(N, 5000.0),
               fanRpm_prev=np.zeros(N))
    return est
