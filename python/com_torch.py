"""
PyTorch port of the COM: Source Files/dynamics_mexfreeNew.m

Same equations, same constants, same order of evaluation as the MATLAB file, written with torch
operations so that gradients can flow from the predicted states back into the 11 γ multipliers.
Every function below has the same name as its MATLAB counterpart, so the two files can be read side by side.

Units (same as MATLAB): temperatures in K, pressures in Pa, SOC in [-], mass flows in kg/s, powers in W.

Shapes: everything is batched. B = batch size (e.g. number of profiles or windows).
    x     [B, 9]   states   (motT, invT, dcdcT, batSoc, batT, compPinPa, compPoutPa, cabIntT, cabAirT)
    u     [B, 6]   inputs   (compNrpm, cabAirMfIn, motPumpNrpm, batPumpNrpm, heaterPwr, radFanrpm)
    est   dict of [B] tensors: estimated / measured parameters (names as in create_dynamics_nmpcNew.m)
    ctrl  dict of [B] tensors: supervisor commands (cmdParallelSerial, cmdBlendAir, cmdExvHeatpump, ...)
    gamma [B, 11]  tuning multipliers p1..p11 (1 = nominal)

Use float64: pressures are ~1e6 Pa and some terms subtract nearly equal enthalpies.

Not ported (they do not affect dx/dt): the evaporator and chiller models (inactive in cold weather, the chiller
only passes the coolant temperature through) and fanRpm_approx (computed but never used in the MATLAB code).
"""
import math
import torch

STATE_NAMES = ["motT", "invT", "dcdcT", "batSoc", "batT", "compPinPa", "compPoutPa", "cabIntT", "cabAirT"]
INPUT_NAMES = ["compNrpm", "cabAirMfIn", "motPumpNrpm", "batPumpNrpm", "heaterPwr", "radFanrpm"]
CTRL_NAMES = ["cmdParallelSerial", "cmdBlendAir", "cmdExvHeatpump", "thermalMode",
              "cmdExvChiller", "cmdExvEvap", "cmdHeater"]
EST_NAMES = [  # every field of estimatedParams that the ported equations read
    "envT", "clntCp", "clntK", "clntRho", "clntMu",
    "cabAirCp", "cabAirK", "cabAirRho", "cabAirMu",
    "innerCondRefCp", "innerCondRefK", "innerCondRefRho", "innerCondRefMu",
    "condEvapRefCp", "condEvapRefK", "condEvapRefRho", "condEvapRefMu",
    "condEvapAirCp", "condEvapAirK", "condEvapAirRho", "condEvapAirMu",
    "speed", "compHin", "compHoutS", "compVnominalIn",
    "refTsatHighPK", "refCpVapHighP", "refHsatLiqHighP", "refTsatLiqHighPK", "refTsatLiqLowPK", "refCpLiqHighP",
    "motQmeas", "invQmeas", "dcdcQmeas", "batImeas",
    "evapVoidFrac", "condVoidFrac", "evapVol_m3", "condVol_m3",
    "evapWallThermalMass_J_K", "condWallThermalMass_J_K",
    "dTesat_dp1", "d_rhoh_l_dp1", "d_rhoh_g_dp1", "dTcsat_dp2", "d_rhoh_l_dp2", "d_rhoh_g_dp2",
]

pi = math.pi
P_REF_SUCTION = 2.0e5   # [Pa] reference inlet pressure for γ12 (flow unchanged at 2 bar)

# ─── COM fixes (21 Nov 2026), OFF by default so the original COM is unchanged. Turn on with set_fixes(True).
# Evidence: heat-flow comparison with the HFM (notes "COM fault analysis"). All are zero-cost: no new states.
#   front_hx    front heat exchanger active (MATLAB overwrote its air flow with 0); conductance fitted to the HFM:
#               G = a*speed[m/s] + b*fan[krpm] + c  [W/K]; q = G (T_amb - T_ref), sign flips by itself.
#   htx         both low-side heat exchangers see evaporating refrigerant at T_sat(p_in); HTX UA × K_HTX.
#   inner_cond  inner condenser: refrigerant at T_sat(p_out), air-side UA fitted to the HFM (IC_UA).
#   heater      all heater power reaches the coolant (was 0.85 / 1.5 = 57 %).
#   cabin       cabin constants refitted (CAB_K); γ1, γ2, γ8 still multiply the same terms.
#   ua          motor / inverter -> coolant UA × K_MOT / K_INV; battery γ7 also used inside the coolant-loop solve.
# New γ columns (optional, 1 = nominal): γ13 HTX UA, γ14 motor UA, γ15 inverter UA.
FIXES = dict(on=False)
FRONT_G = (7.88, 20.09, 70.5)
CAB_K = (1 / 0.01611, 0.3056, 0.4513, 1.4400)   # cabin (air speed, air<->interior, interior speed, interior<->ambient):
                                  # fitted on the 18 training runs, 2-state cabin free run driven by the fixed condenser.
                                  # Air speed = 1/alpha_dt_air_num: the cabin air then has its real heat capacity (3 m³).
                                  # 1-h test MAE cabin air 19.3 -> 0.75 K, interior 6.1 -> 0.52 K at γ = 1.
IC_UA = (190.7, 0.669)            # inner condenser air-side UA = a * mdot_air^b [W/K]; fit WLTCHW −20 °C (q within 1 %),
                                  # −12 °C (not used in fit) within 6 %     # fit on WLTCHW −20 °C (R² 0.85); −12 °C (not used in fit) gives 7.06, 21.6, 67.8
K_HTX = 2.6159; K_MOT = 1.5887; K_INV = 5.1496 * 25   # UA multipliers at γ = 1 (calibrated; same as MATLAB com_fix_constants)
N_GAMMA_FIX = 15
# γ rescaling (fixes on): the trained network kept these γ at their 0.1 / 10 limits, so the constant behind each is
# moved to where the network wanted it and γ = 1 now means that value (range still ×/÷ 10 around it).
# Values from the γ distribution of the round-3 network on the training runs (notes, 21 Nov evening).
GAMMA_SCALE = {4: 0.3, 7: 0.03, 9: 4.0, 10: 0.28, 11: 6.0}   # γ4 inverter speed, γ7 battery UA, γ9 low-side heat,
                                                              # γ10 p_out speed, γ11 p_in speed
# γ10 needs a wider range than ×/÷ 10: ≈ 1-8 during the compressor start-up (p_out 2 -> 35 bar in ~20 s) and ≈ 0.015-0.1
# afterwards. Effective γ10 = 0.28 γ^1.5 -> 0.009 ... 8.9 for γ in [0.1, 10].
GAMMA_POWER = {10: 1.5}


def set_fixes(on=True, k_htx=None, k_mot=None, k_inv=None):
    """Turn the COM fixes on/off. Optional calibration constants (UA multipliers at γ = 1)."""
    global K_HTX, K_MOT, K_INV
    FIXES["on"] = bool(on)
    if k_htx is not None: K_HTX = float(k_htx)
    if k_mot is not None: K_MOT = float(k_mot)
    if k_inv is not None: K_INV = float(k_inv)
sqrt, exp, tanh, log = torch.sqrt, torch.exp, torch.tanh, torch.log


class _S(dict):
    """Small helper so the code can write cmp.mot.clntToutK like the MATLAB structs (cmp.mot.output.clntToutK)."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v


# ═══════════════════════════════════════════════════════════════════ main function
def com_xdot(x, u, est, ctrl, gamma, return_outputs=False):
    """dx/dt of the COM. Mirrors dynamics_mexfreeNew(x, u, p)."""
    P = [None] + [gamma[:, i] for i in range(gamma.shape[1])]   # P[1]..P[11] (+ optional P[12]), 1-based like MATLAB
    if len(P) < 13:                                             # no γ12 given -> 0 = original fixed suction density
        P.append(torch.zeros_like(gamma[:, 0]))
    while len(P) < 16:                                          # no γ13..γ15 given -> 1 (used only with fixes on)
        P.append(torch.ones_like(gamma[:, 0]))
    if FIXES["on"]:
        for i, k in GAMMA_SCALE.items():
            P[i] = k * P[i] ** GAMMA_POWER.get(i, 1.0)

    s = _S(motT=x[:, 0], invT=x[:, 1], dcdcT=x[:, 2], batSoc=x[:, 3], batT=x[:, 4],
           compPinPa=x[:, 5], compPoutPa=x[:, 6], cabIntT=x[:, 7], cabAirT=x[:, 8])
    uu = _S(compNrpm=u[:, 0], cabAirMfIn=u[:, 1] + 1e-3, motPumpNrpm=u[:, 2], batPumpNrpm=u[:, 3],
            heaterPwr=u[:, 4], radFanrpm=u[:, 5])
    uu.batImeas = est["batImeas"] + 1e-3
    e, c = _S(est), _S(ctrl)

    cmp = _S()
    pumpMdl(s, uu, e, c, cmp)
    compressorMdl(s, uu, e, c, cmp, P)
    innerCondenserMdl(s, uu, e, c, cmp)
    hpExvMdl(s, uu, e, c, cmp)
    condEvapMdl(s, uu, e, c, cmp)

    uu.htxClntTinK = solve_loop_htx_in_explicit(s, uu, e, c, cmp, P)   # coolant loop closed in one step

    htxMdl(s, uu, e, c, cmp, P)
    cmp.chiller = _S(clntToutK=cmp.htx.clntToutK)                   # chillerMdl: output = input (inactive)
    heaterMdl(s, uu, e, c, cmp)

    batMdl(s, uu, e, c, cmp, P)
    dcdcMdl(s, uu, e, c, cmp, P)
    motMdl(s, uu, e, c, cmp, P)
    invMdl(s, uu, e, c, cmp, P)
    cabMdl(s, uu, e, c, cmp, P)

    pressureStateCalcMdl(s, uu, e, c, cmp, P)

    xdot = torch.stack([cmp.mot.motT_dt, cmp.inv.invT_dt, cmp.dcdc.dcdcT_dt,
                        cmp.bat.batSoc_dt, cmp.bat.batT_dt,
                        cmp.pressure.compPin_dt, cmp.pressure.compPout_dt,
                        cmp.cab.cabIntT_dt, cmp.cab.cabAirT_dt], dim=1)
    if not return_outputs:
        return xdot
    pwrEstimation(s, uu, e, c, cmp)
    outputs = {
        "ventAirT_K": cmp.innerCond.cabAirToutK,      # air after inner condenser
        "invClntTout_K": cmp.inv.clntToutK,
        "motClntTin_K": cmp.mot.clntTinK,
        "heaterClntTout_K": cmp.heater.clntToutK,
        "batClntTout_K": cmp.bat.clntToutK,
        "htxClntTout_K": cmp.htx.clntToutK,
        "compTout_K": cmp.compressor.compToutK,
        "refMf": cmp.compressor.refMf,
        "pwrCompElec": cmp.pwrEst.pwrCompElec,
        "pwrTem": cmp.pwrEst.pwrTem,                 # total thermal-management power (the NMPC cost J)
    }
    return xdot, outputs


# ═══════════════════════════════════════════════════════════════════ coolant loop solver
def solve_loop_htx_in_explicit(s, uu, e, c, cmp, P=None):
    fix = FIXES["on"] and P is not None
    clntCp = e.clntCp
    m_mot = cmp.motPump.clntMf
    m_tot_bat = cmp.batPump.clntMf
    split = 0.11
    m_dcdc = split * m_tot_bat
    m_bat = m_tot_bat - m_dcdc

    a_htx, b_htx = local_htx_affine(cmp, e, m_mot, (K_HTX * P[13]) if fix else None)
    a_heat, b_heat = local_heater_affine(e, m_tot_bat, clntCp, uu)

    a_T2_T0 = a_heat * a_htx
    b_T2 = a_heat * b_htx + b_heat

    a_bat, b_bat = local_branch_affine_bat(e, m_bat, clntCp, s.batT, P[7] if fix else None)
    a_dcdc, b_dcdc = local_branch_affine_dcdc(m_dcdc, clntCp, s.dcdcT)

    w_bat = m_bat / (m_bat + m_dcdc + 1e-12)
    w_dcdc = 1 - w_bat
    a_mix = w_bat * a_bat + w_dcdc * a_dcdc
    b_mix = w_bat * b_bat + w_dcdc * b_dcdc

    cmdPS = c.cmdParallelSerial
    a_Tmotin_T0 = cmdPS * (a_mix * a_T2_T0) + (1 - cmdPS) * a_htx
    b_Tmotin = cmdPS * (a_mix * b_T2 + b_mix) + (1 - cmdPS) * b_htx

    k_mot = (K_MOT * P[14]) if fix else 1.0
    k_inv = (K_INV * P[15]) if fix else 1.0
    a_mot, b_mot = local_component_affine(e, m_mot, clntCp, s.motT, convection=6.0 * k_mot, ntu_factor=1.0)
    a_inv, b_inv = local_component_affine(e, m_mot, clntCp, s.invT, convection=6.0 * k_inv, ntu_factor=0.1)

    a_chain = a_mot * a_Tmotin_T0
    b_chain = a_mot * b_Tmotin + b_mot

    alpha = a_inv * a_chain
    beta = a_inv * b_chain + b_inv
    return beta / (1 - alpha + 1e-6)


def _dittus_boelter_UA(mf, area, rho, mu, cp, k, D, re_eps=0.0):
    """Shared pattern: velocity -> Re -> Pr -> Nu = 0.023 Re^0.8 Pr^0.33 -> h."""
    V = mf / (rho * area + 1e-6)
    Re = (rho * V * D) / (mu + 1e-6)
    Pr = (cp * mu) / (k + 1e-6)
    Nu = 0.023 * ((Re + re_eps) ** 0.8) * (Pr ** 0.33)
    return (Nu * k) / (D + 1e-6)


def _htx_ref_tin(cmp):
    """Refrigerant temperature entering the HTX. Fixes on: evaporating refrigerant at T_sat(p_in) (= EXV outlet),
    not the front-HX outlet computed as single-phase vapour."""
    return cmp.hpExv.refToutK if FIXES["on"] else cmp.condEvap.refToutK


def local_htx_affine(cmp, e, m_mot, k_ua=None):
    refArea = 7.854e-05; clntArea = 0.00028353
    totalHxArea = 64 * pi * 0.0035 * 0.1
    ntuCorrectionFactor = 0.9
    NTU_MAX = 10
    refMf = cmp.compressor.refMf
    clntMf = m_mot
    refTinK = _htx_ref_tin(cmp)
    refD = math.sqrt((4 * refArea) / pi); clntD = math.sqrt((4 * clntArea) / pi)
    h_ref = _dittus_boelter_UA(refMf, refArea, e.condEvapRefRho, e.condEvapRefMu, e.condEvapRefCp, e.condEvapRefK, refD)
    h_clnt = _dittus_boelter_UA(clntMf, clntArea, e.clntRho, e.clntMu, e.clntCp, e.clntK, clntD)
    thermalResistance = (1 / (h_ref * totalHxArea + 1e-6)) + (1 / (h_clnt * totalHxArea + 1e-6))
    UA = 1 / (thermalResistance + 1e-6)
    if k_ua is not None:
        UA = UA * k_ua
    c_clnt = clntMf * e.clntCp
    NTU_raw = (ntuCorrectionFactor * UA) / (c_clnt + 1e-6)
    NTU = NTU_MAX * tanh(NTU_raw / NTU_MAX)
    A = 1 - exp(-NTU)
    return 1 - A, A * refTinK


def local_heater_affine(e, m_tot_bat, clntCp, uu):
    heaterEfficiency, heatTransferCorrection = (1.0, 1.0) if FIXES["on"] else (0.85, 1.5)
    heaterActualPowerW = heaterEfficiency * uu.heaterPwr
    denominator = heatTransferCorrection * m_tot_bat * clntCp
    dT = heaterActualPowerW / (denominator + 1e-6)
    return torch.ones_like(dT), dT


def _branch_S(UA, ntu_factor, c_clnt, k_cond):
    """effectiveness (NTU-capped) + capped conduction term, used by all coolant branches."""
    NTU_MAX = 10
    NTU_raw = (ntu_factor * UA) / (c_clnt + 1e-6)
    NTU = NTU_MAX * tanh(NTU_raw / NTU_MAX)
    effectiveness = 1 - exp(-NTU)
    k_cond_raw = k_cond / (c_clnt + 1e-6)
    k_cond_eff = NTU_MAX * tanh(k_cond_raw / NTU_MAX)
    return effectiveness + k_cond_eff


def local_branch_affine_bat(e, m_bat, clntCp, T_batK, g7=None):
    clntChannelD = 0.0092; hxAreaFactor = 0.736
    convectionTuningFactor = 200; conductionTuningFactor = 0.7
    h_conv = 3.66 * e.clntK / clntChannelD
    UA = h_conv * hxAreaFactor * convectionTuningFactor
    # batMdl uses NTU = γ7 UA / c (no 0.1); with fixes the loop solve uses the same, so both agree
    ntu = 0.1 if g7 is None else g7
    S = _branch_S(UA, ntu, m_bat * clntCp, conductionTuningFactor * hxAreaFactor / clntChannelD)
    return 1 - S, S * T_batK


def local_branch_affine_dcdc(m_dcdc, clntCp, T_dcdcK):
    clntChannelD = 0.0092; hxAreaFactor = 0.736
    convectionTuningFactor = 1; conductionTuningFactor = 0.245
    h_conv = 3.66 * 0.254 / clntChannelD                       # fixed clntK = 0.254 here (as in MATLAB)
    UA = h_conv * hxAreaFactor * convectionTuningFactor
    S = _branch_S(UA, 1.0, m_dcdc * clntCp, conductionTuningFactor * hxAreaFactor / clntChannelD)
    return 1 - S, S * T_dcdcK


def local_component_affine(e, m_mot, clntCp, T_K, convection, ntu_factor):
    """local_component_affine_mot (ntu_factor 1.0) and _inv (ntu_factor 0.1): otherwise identical."""
    clntChannelD = 0.0092; hxAreaFactor = 0.736
    conductionTuningFactor = 0.254
    h_conv = 3.66 * e.clntK / clntChannelD
    UA = h_conv * hxAreaFactor * convection
    S = _branch_S(UA, ntu_factor, m_mot * clntCp, conductionTuningFactor * hxAreaFactor / clntChannelD)
    return 1 - S, S * T_K


# ═══════════════════════════════════════════════════════════════════ components
def htxMdl(s, uu, e, c, cmp, P=None):
    refArea = 7.854e-05; clntArea = 0.00028353
    totalHxArea = 64 * pi * 0.0035 * 0.1
    ntuCorrectionFactor = 0.9
    NTU_MAX = 10
    clntMf = cmp.motPump.clntMf
    refMf = cmp.compressor.refMf
    refTinK = _htx_ref_tin(cmp)
    clntTinK = uu.htxClntTinK
    refD = math.sqrt((4 * refArea) / pi); clntD = math.sqrt((4 * clntArea) / pi)
    h_ref = _dittus_boelter_UA(refMf, refArea, e.condEvapRefRho, e.condEvapRefMu, e.condEvapRefCp, e.condEvapRefK, refD)
    h_clnt = _dittus_boelter_UA(clntMf, clntArea, e.clntRho, e.clntMu, e.clntCp, e.clntK, clntD)
    thermalResistance = (1 / (h_ref * totalHxArea + 1e-6)) + (1 / (h_clnt * totalHxArea + 1e-6))
    UA = 1 / (thermalResistance + 1e-6)
    if FIXES["on"] and P is not None:
        UA = UA * K_HTX * P[13]
    c_clnt = clntMf * e.clntCp
    c_min = clntMf * e.clntCp
    NTU_raw = (ntuCorrectionFactor * UA) / (c_min + 1e-6)
    NTU = NTU_MAX * tanh(NTU_raw / NTU_MAX)
    effectiveness = 1 - exp(-NTU)
    qMax = c_min * (clntTinK - refTinK)
    qActual = effectiveness * qMax
    clntToutK = clntTinK - qActual / (c_clnt + 1e-6)
    cmp.htx = _S(clntToutK=clntToutK, refToutK=refTinK, qActualW=qActual)


def cabMdl(s, uu, e, c, cmp, P):
    glassArea_m2 = 3.5; glassU_W_m2K = 40; glassThick_m = 0.002; glassK_W_mK = 0.96
    glassMass_kg = 36.4; glassCp_J_kgK = 840
    doorsArea_m2 = 4.0; doorsU_W_m2K = 40; doorsThick_m = 0.03; doorsK_W_mK = 0.08
    doorsMass_kg = 75.0; doorsCp_J_kgK = 500
    roofArea_m2 = 2.0; roofU_W_m2K = 40; roofThick_m = 0.02; roofK_W_mK = 0.08
    roofMass_kg = 30.0; roofCp_J_kgK = 500
    cabVolume_m3 = 3.0; airDensity_kg_m3 = 1.225; qHuman_W = 70.0
    airCp_J_kgK = 1006
    alpha_R_glass = 0.01217; alpha_R_doors = 0.15076; alpha_R_roof = 0.63668
    alpha_dt_int_num = 42237.0; alpha_dt_int_den = 2287.85
    alpha_dt_air_num = 0.01611; alpha_R_air2int = 0.02475
    alpha_Q_in2out = 1.0; alpha_Q_int2air = 1.0

    cabAirT_K = s.cabAirT
    cabIntT_K = s.cabIntT
    airMf_In = uu.cabAirMfIn
    airTempVent_K = cmp.innerCond.cabAirToutK
    envT_K = e.envT

    R_glass = alpha_R_glass * (1 / (glassU_W_m2K * glassArea_m2) + glassThick_m / (glassK_W_mK * glassArea_m2))
    R_doors = alpha_R_doors * (1 / (doorsU_W_m2K * doorsArea_m2) + doorsThick_m / (doorsK_W_mK * doorsArea_m2))
    R_roof = alpha_R_roof * (1 / (roofU_W_m2K * roofArea_m2) + roofThick_m / (roofK_W_mK * roofArea_m2))
    R_total_env2int = R_glass + R_doors + R_roof
    totalMass_int = glassMass_kg + doorsMass_kg + roofMass_kg
    avgCp_int = (glassMass_kg * glassCp_J_kgK + doorsMass_kg * doorsCp_J_kgK + roofMass_kg * roofCp_J_kgK) / totalMass_int
    thermalMass_int = totalMass_int * avgCp_int

    kA, kC, kI, kE = CAB_K if FIXES["on"] else (1.0, 1.0, 1.0, 1.0)
    q_env2int = kE * (envT_K - cabIntT_K) / (R_total_env2int + 1e-6)
    q_air2int = alpha_Q_in2out * (cabAirT_K - cabIntT_K) / (R_total_env2int + 1e-6)
    qNet_int = q_env2int + q_air2int
    cabIntT_dt = (kI * P[1] * alpha_dt_int_num * qNet_int) / (alpha_dt_int_den * thermalMass_int + 1e-6)

    thermalMass_air = cabVolume_m3 * airDensity_kg_m3 * airCp_J_kgK
    qHvac = airMf_In * airCp_J_kgK * (airTempVent_K - cabAirT_K)
    R_int2air = alpha_R_air2int * R_total_env2int
    q_int2air = kC * alpha_Q_int2air * (cabIntT_K - cabAirT_K) / (R_int2air + 1e-6)
    cabAirT_dt = (kA * P[2] * alpha_dt_air_num / (thermalMass_air + 1e-6)) * (qHvac + qHuman_W + P[8] * q_int2air)
    cmp.cab = _S(cabIntT_dt=cabIntT_dt, cabAirT_dt=cabAirT_dt)


def pumpMdl(s, uu, e, c, cmp):
    pumpVolEff = 0.92; pumpDispLperRev = 0.02; densityCorrection = 1.0660
    clntRho = 1100
    pumpDispM3perRev = pumpDispLperRev * 1e-3
    batClntMf = clntRho * densityCorrection * pumpDispM3perRev * (uu.batPumpNrpm / 60) * pumpVolEff + 1e-4
    motClntMf = clntRho * densityCorrection * pumpDispM3perRev * (uu.motPumpNrpm / 60) * pumpVolEff + 1e-4
    cmp.motPump = _S(clntMf=motClntMf)
    cmp.batPump = _S(clntMf=batClntMf)


def compressorMdl(s, uu, e, c, cmp, P=None):
    compDispM3perRev = 80e-6; isenEff = 0.65; mechEff = 0.9; massFlowCorrection = 0.12
    compPinPa = s.compPinPa
    compPoutPa = s.compPoutPa
    compNrpm = uu.compNrpm + 1e-5
    h_in = e.compHin
    h_outS = e.compHoutS
    vNominalIn = e.compVnominalIn

    p_in_floor = 0.3e5; eps_pFloor = 1e6
    compPinSafe = (compPinPa + p_in_floor + sqrt((compPinPa - p_in_floor) ** 2 + eps_pFloor)) / 2   # smooth max(p, 0.3 bar)
    # γ12: suction specific volume follows the inlet pressure, v_in = v_nom (p_ref / p_in)^γ12 (p_ref = 2 bar).
    # γ12 = 0 is the original COM (density fixed at 3 bar, 11 °C); real R134a vapour here is ≈ 0.8-0.9.
    if P is not None and P[12] is not None:
        vNominalIn = vNominalIn * (P_REF_SUCTION / compPinSafe) ** P[12]
    pressureRatio = compPoutPa / compPinSafe
    volEff_calc = -0.0286 * pressureRatio + 1.0286
    k_smooth = 50
    # MATLAB: 0.01 + (1/k) log(1 + exp(k (v - 0.01)))  -> written with logaddexp so it cannot overflow
    z = k_smooth * (volEff_calc - 0.01)
    volEff = 0.01 + torch.logaddexp(torch.zeros_like(z), z) / k_smooth

    angularVel = compNrpm * (2 * pi / 60)
    refMf = volEff * angularVel * compDispM3perRev * massFlowCorrection / vNominalIn + 1e-4
    h_out = h_in + (h_outS - h_in) / isenEff
    temp_rise = 2 * ((h_outS - h_in) / (e.refCpVapHighP + 1e-6))
    compToutK = e.refTsatHighPK + temp_rise
    powerW = refMf * (h_out - h_in) / mechEff
    cmp.compressor = _S(refMf=refMf, compToutK=compToutK, h_out=h_out, h_in=h_in, powerW=powerW)


def innerCondenserMdl(s, uu, e, c, cmp):
    refArea = 7.854e-05; airArea = 0.04
    totalHxArea = 1.1088 + 0.9952
    wallThermalResistance = 4.1868e-07
    ntuAreaFactor = 18 * 40
    NTU_MAX = 10
    refMf = cmp.compressor.refMf
    cabAirMfIn = uu.cabAirMfIn
    # fixes on: the refrigerant condenses at T_sat(p_out) (the 120 °C discharge temperature only holds for the
    # short desuperheating part), and the air-side UA is fitted to the HFM (IC_UA). No cap on q: the heat must
    # depend on T_sat(p_out) - T_air, that is what pulls p_out back (a cap made p_out neutrally stable).
    refTinK = e.refTsatHighPK if FIXES["on"] else cmp.compressor.compToutK
    cabAirTinK = s.cabAirT
    cmdBlendAir = c.cmdBlendAir
    refD = 0.0100; airD = 0.0190

    refV = refMf / (e.innerCondRefRho * refArea + 1e-6)
    airV = (cmdBlendAir * cabAirMfIn / (e.cabAirRho * airArea + 1e-6)) + 1e-6
    refRe = (e.innerCondRefRho * refV * refD) / (e.innerCondRefMu + 1e-6)
    airRe = (e.cabAirRho * airV * airD) / (e.cabAirMu + 1e-6)
    refPr = (e.innerCondRefCp * e.innerCondRefMu) / (e.innerCondRefK + 1e-6)
    airPr = (e.cabAirCp * e.cabAirMu) / (e.cabAirK + 1e-6)
    refNu = 0.023 * (refRe ** 0.8) * (refPr ** 0.33)
    airNu = 0.023 * (airRe ** 0.8) * (airPr ** 0.33)
    refU = (refNu * e.innerCondRefK) / (refD + 1e-6)
    airU = (airNu * e.cabAirK) / (airD + 1e-6)
    thermalResistance = (1 / (refU * refArea + 1e-6)) + (1 / (airU * airArea + 1e-6)) + wallThermalResistance

    c_air = cabAirMfIn * e.cabAirCp
    c_min = c_air
    sigmoid_gate_c_air = 0.5 * (1 + tanh(5e4 * (cabAirMfIn - 1e-4)))
    NTU_raw = (ntuAreaFactor * totalHxArea) / (thermalResistance * c_min + 1e-6)
    NTU_raw = NTU_raw * sigmoid_gate_c_air
    NTU = NTU_MAX * tanh(NTU_raw / NTU_MAX)
    effectiveness = 1 - exp(-NTU)

    airTinK = cabAirTinK
    qMax = c_min * (refTinK - airTinK)
    qActual = effectiveness * qMax
    if FIXES["on"]:
        a_ua, b_ua = IC_UA
        UA_ic = a_ua * (cabAirMfIn + 1e-6) ** b_ua
        effectiveness = 1 - exp(-UA_ic / (c_air + 1e-6))
        qActual = effectiveness * qMax * sigmoid_gate_c_air
    cabAirToutK = cabAirTinK + (qActual / (c_air + 1e-6)) * sigmoid_gate_c_air

    c_ref = refMf * e.innerCondRefCp
    sigmoid_gate_c_ref = 0.5 * (1 + tanh(5e4 * (refMf - 1e-4)))
    refToutK = refTinK - (0.35 * qActual / (c_ref + 1e-6)) * sigmoid_gate_c_ref
    h_subcooled = e.refHsatLiqHighP + e.innerCondRefCp * (refToutK - e.refTsatLiqHighPK)
    cmp.innerCond = _S(cabAirToutK=cabAirToutK, refToutK=refToutK, qActualW=qActual, h_out=h_subcooled, airTinK=airTinK)


def hpExvMdl(s, uu, e, c, cmp):
    refToutK_cond = cmp.innerCond.refToutK
    h_comp_out = cmp.compressor.h_out
    cmdExvHeatpump = c.cmdExvHeatpump
    h_subcooled_liquid = e.refHsatLiqHighP + e.refCpLiqHighP * (refToutK_cond - e.refTsatLiqHighPK)
    refToutK_expanded = e.refTsatLiqLowPK
    h_out = h_subcooled_liquid * cmdExvHeatpump + h_comp_out * (1 - cmdExvHeatpump)
    refToutK_out = refToutK_expanded * cmdExvHeatpump + refToutK_cond * (1 - cmdExvHeatpump)
    cmp.hpExv = _S(refToutK=refToutK_out, h_out=h_out)


def condEvapMdl(s, uu, e, c, cmp):
    refArea = 7.854e-05; airArea = 0.04
    totalHxArea = 1.1088 + 0.9952
    NTU_MAX = 10
    refMf = cmp.compressor.refMf
    refTinK = cmp.hpExv.refToutK
    # MATLAB computes an air flow from vehicle speed and fan speed, then overwrites it with airMf = 0,
    # so the front heat exchanger exchanges no heat. Kept exactly like that.
    airMf = torch.zeros_like(refMf)
    airTinK = e.envT
    thermalMode = c.thermalMode

    refD = math.sqrt((4 * refArea) / pi); airD = math.sqrt((4 * airArea) / pi)
    h_ref = _dittus_boelter_UA(refMf, refArea, e.condEvapRefRho, e.condEvapRefMu, e.condEvapRefCp, e.condEvapRefK, refD, re_eps=1e-4)
    h_air = _dittus_boelter_UA(airMf, airArea, e.condEvapAirRho, e.condEvapAirMu, e.condEvapAirCp, e.condEvapAirK, airD, re_eps=1e-4)
    thermalResistance = (1 / (h_ref * totalHxArea + 1e-6)) + (1 / (h_air * totalHxArea + 1e-6))
    UA = 1 / (thermalResistance + 1e-6)

    c_ref = refMf * e.condEvapRefCp
    c_air = airMf * e.condEvapAirCp
    isHeatPumpMode = 1 / (1 + exp(-10 * (thermalMode - 1)))
    if FIXES["on"]:
        # evaporating refrigerant at T_sat(p_in): q = G (T_amb - T_ref), G fitted to the HFM (see FRONT_G).
        # q > 0: heat taken from the outside air; q < 0 (refrigerant warmer than ambient): heat given to it.
        a, b, c0 = FRONT_G
        G = a * e.speed / 3.6 + b * uu.radFanrpm / 1000 + c0
        qActual = isHeatPumpMode * G * (airTinK - refTinK)
        cmp.condEvap = _S(refToutK=refTinK, airToutK=airTinK - qActual / (G + 1e-6), qActualW=qActual)
        return
    qMax_evap = c_air * (airTinK - refTinK)
    qMax_cond = c_air * (refTinK - airTinK)
    qMax = qMax_evap * isHeatPumpMode + qMax_cond * (1 - isHeatPumpMode)
    NTU_raw = UA / (c_air + 1e-6)
    NTU = NTU_MAX * tanh(NTU_raw / NTU_MAX)
    effectiveness = 1 - exp(-NTU)
    qActual = effectiveness * qMax
    refDir = isHeatPumpMode - (1 - isHeatPumpMode)
    refToutK = refTinK + refDir * qActual / (c_ref + 1e-6)
    airToutK = airTinK - refDir * qActual / (c_air + 1e-6)
    cmp.condEvap = _S(refToutK=refToutK, airToutK=airToutK, qActualW=qActual)


def heaterMdl(s, uu, e, c, cmp):
    heaterEfficiency, heatTransferCorrection = (1.0, 1.0) if FIXES["on"] else (0.85, 1.5)
    clntTinK = cmp.chiller.clntToutK
    clntMf = cmp.batPump.clntMf
    delta_T = heaterEfficiency * uu.heaterPwr / (heatTransferCorrection * clntMf * e.clntCp + 1e-6)
    cmp.heater = _S(clntToutK=clntTinK + delta_T, clntTinK=clntTinK)


def batMdl(s, uu, e, c, cmp, P):
    clntChannelD = 0.0092; hxAreaFactor = 0.736; NTU_MAX = 10
    batMassKg = 2.5 * 20 * 4
    batCp = 795
    convectionTuningFactor = 200; conductionTuningFactor = 0.7
    jouleHeatingTuningFactor = 70
    socCapacityFactor = 44 * 20
    thermalMassTuningFactor = 0.75
    dcdcFlowSplitRatio = 0.11

    batT_K = s.batT
    batSoc = s.batSoc
    batI = uu.batImeas
    clntTinK = cmp.heater.clntToutK
    clntCp = e.clntCp
    clntTotalMf = cmp.batPump.clntMf
    clntMf_bat = clntTotalMf - dcdcFlowSplitRatio * clntTotalMf

    h_conv = 3.66 * e.clntK / clntChannelD
    UA = h_conv * hxAreaFactor * convectionTuningFactor
    c_clnt = clntMf_bat * clntCp
    NTU_raw = (P[7] * UA) / (c_clnt + 1e-6)
    NTU = NTU_MAX * tanh(NTU_raw / NTU_MAX)
    effectiveness = 1 - exp(-NTU)
    qConv = effectiveness * c_clnt * (batT_K - clntTinK)
    qCond = (conductionTuningFactor * hxAreaFactor / clntChannelD) * (batT_K - clntTinK)
    qTotal = qConv + qCond

    capacityAh = 0.0075 * batT_K ** 2 - 4.5401 * batT_K + 962.67
    X, Y = batT_K, batSoc
    resistanceOhm_poly = (0.5345 - 0.00348 * X - 0.02749 * Y + 5.748e-06 * X ** 2 + 7.589e-05 * X * Y
                          + 0.02773 * Y ** 2 + 9.909e-08 * X ** 2 * Y - 0.0001321 * X * Y ** 2 + 0.008301 * Y ** 3)
    r_floor = 1e-8; eps_res = 1e-18
    resistanceOhm = (resistanceOhm_poly + r_floor + sqrt((resistanceOhm_poly - r_floor) ** 2 + eps_res)) / 2

    batSoc_dt = -batI / (capacityAh * 4 * socCapacityFactor + 1e-6)
    qJoule = jouleHeatingTuningFactor * batI ** 2 * resistanceOhm
    thermalMass = thermalMassTuningFactor * batMassKg * batCp
    batT_dt = P[3] * (0.98 * qJoule - qTotal) / (thermalMass + 1e-6)
    clntToutK = clntTinK + qTotal / (c_clnt + 1e-6)
    cmp.bat = _S(batT_dt=batT_dt, batSoc_dt=batSoc_dt, clntToutK=clntToutK, clntMf=clntMf_bat, clntTinK=clntTinK)


def dcdcMdl(s, uu, e, c, cmp, P):
    clntChannelD = 0.0092; hxAreaFactor = 0.736; NTU_MAX = 10
    dcdcMassKg = 1.0; dcdcCp = 447
    convectionTuningFactor = 1; conductionTuningFactor = 0.245
    thermalMassTuningFactor = 0.1
    flowSplitRatio = 0.11

    dcdcT_K = s.dcdcT
    clntTinK = cmp.heater.clntToutK
    clntCp = e.clntCp
    clntMf = flowSplitRatio * cmp.batPump.clntMf
    qGenerated = e.dcdcQmeas

    h_conv = 3.66 * 0.254 / clntChannelD
    UA = h_conv * hxAreaFactor * convectionTuningFactor
    c_clnt = clntMf * clntCp
    C0 = 1e-4 * clntCp
    gate = 1 / (1 + exp(-500 * (c_clnt - C0)))
    NTU_raw = UA / (c_clnt + 1e-6) * gate
    NTU = NTU_MAX * tanh(NTU_raw / NTU_MAX)
    effectiveness = 1 - exp(-NTU)
    qConv = effectiveness * c_clnt * (dcdcT_K - clntTinK)
    qCond = (conductionTuningFactor * hxAreaFactor / clntChannelD) * (dcdcT_K - clntTinK)
    qRemoved = qConv + qCond
    thermalMass = thermalMassTuningFactor * dcdcMassKg * dcdcCp
    dcdcT_dt = 0.05 * (qGenerated - qRemoved) / (thermalMass + 1e-6)      # note: no γ here (fixed 0.05)
    clntToutK = clntTinK + (qRemoved / (c_clnt + 1e-6)) * gate
    cmp.dcdc = _S(dcdcT_dt=dcdcT_dt, clntToutK=clntToutK, clntMf=clntMf)


def invMdl(s, uu, e, c, cmp, P):
    clntChannelD = 0.0092; hxAreaFactor = 0.736; NTU_MAX = 10
    invMassKg = 1.0; invCp = 447
    convectionTuningFactor = 6.0 * (K_INV * P[15] if FIXES["on"] else 1.0); conductionTuningFactor = 0.254

    invT_K = s.invT
    clntCp = e.clntCp
    clntMf = cmp.motPump.clntMf
    qGenerated = e.invQmeas
    clntTinK = cmp.mot.clntToutK

    h_conv = 3.66 * e.clntK / clntChannelD
    UA = h_conv * hxAreaFactor * convectionTuningFactor
    c_clnt = clntMf * clntCp
    NTU_raw = (0.1 * UA) / (c_clnt + 1e-6)
    NTU = NTU_MAX * tanh(NTU_raw / NTU_MAX)
    effectiveness = 1 - exp(-NTU)
    qConv = effectiveness * c_clnt * (invT_K - clntTinK)
    qCond = (conductionTuningFactor * hxAreaFactor / clntChannelD) * (invT_K - clntTinK)
    qRemoved = qConv + qCond
    invT_dt = P[4] * (qGenerated - qRemoved) / (invMassKg * invCp + 1e-6)
    clntToutK = clntTinK + qRemoved / (c_clnt + 1e-6)
    cmp.inv = _S(invT_dt=invT_dt, clntToutK=clntToutK)


def pressureStateCalcMdl(s, uu, e, c, cmp, P):
    q_evap = cmp.htx.qActualW + cmp.condEvap.qActualW
    q_cond = cmp.innerCond.qActualW
    refMf = cmp.compressor.refMf
    h1 = cmp.compressor.h_in; h2 = cmp.compressor.h_out
    h3 = cmp.innerCond.h_out; h4 = cmp.innerCond.h_out          # placeholder in MATLAB too
    ye, yc = e.evapVoidFrac, e.condVoidFrac
    Ve, Vc = e.evapVol_m3, e.condVol_m3
    Mwe_ce, Mwc_cc = e.evapWallThermalMass_J_K, e.condWallThermalMass_J_K

    f1 = P[9] * q_evap + refMf * (h4 - h1)
    D11 = Ve * ((1 - ye) * e.d_rhoh_l_dp1 + ye * e.d_rhoh_g_dp1 - 1 + (Mwe_ce / Ve) * e.dTesat_dp1)
    f2 = -q_cond + P[6] * refMf * (h2 - h3)
    D22 = Vc * ((1 - yc) * e.d_rhoh_l_dp2 + yc * e.d_rhoh_g_dp2 - 1 + (Mwc_cc / Vc) * e.dTcsat_dp2)

    k_gate = 100
    gate_D11 = 1 / (1 + exp(-k_gate * (torch.abs(D11) - 1e-4)))
    gate_D22 = 1 / (1 + exp(-k_gate * (torch.abs(D22) - 1e-4)))
    compPin_dt = (P[11] * f1 / (500 * D11 + 1e-6)) * gate_D11
    compPout_dt = (P[10] * f2 / (1 * D22 + 1e-6)) * gate_D22
    cmp.pressure = _S(compPin_dt=compPin_dt, compPout_dt=compPout_dt)


def motMdl(s, uu, e, c, cmp, P):
    clntChannelD = 0.0092; hxAreaFactor = 0.736
    motMassKg = 46.0; motCp = 447
    convectionTuningFactor = 6.0 * (K_MOT * P[14] if FIXES["on"] else 1.0); conductionTuningFactor = 0.254
    NTU_MAX = 10

    motT_K = s.motT
    clntCp = e.clntCp
    clntMf = cmp.motPump.clntMf
    qGenerated = e.motQmeas
    batClntToutK, batClntMf = cmp.bat.clntToutK, cmp.bat.clntMf
    dcdcClntToutK, dcdcClntMf = cmp.dcdc.clntToutK, cmp.dcdc.clntMf
    mixedBatLoopToutK = (batClntToutK * batClntMf + dcdcClntToutK * dcdcClntMf) / (batClntMf + dcdcClntMf + 1e-12)
    cmdPS = c.cmdParallelSerial
    clntTinK = cmdPS * mixedBatLoopToutK + (1 - cmdPS) * cmp.htx.clntToutK

    h_conv = 3.66 * e.clntK / clntChannelD
    UA = h_conv * hxAreaFactor * convectionTuningFactor
    c_clnt = clntMf * clntCp
    gate = 0.5 * (1 + tanh(1000 * (clntMf - 1e-3)))
    NTU_raw = (UA / (c_clnt + 1e-6)) * gate
    NTU = NTU_MAX * tanh(NTU_raw / NTU_MAX)
    effectiveness = 1 - exp(-NTU)
    qConv = effectiveness * c_clnt * (motT_K - clntTinK)
    qCond = (conductionTuningFactor * hxAreaFactor / clntChannelD) * (motT_K - clntTinK)
    qRemoved = qConv + qCond
    motT_dt = P[5] * (qGenerated - qRemoved) / (motMassKg * motCp + 1e-6)
    clntToutK = clntTinK + (qRemoved / (c_clnt + 1e-6)) * gate
    cmp.mot = _S(motT_dt=motT_dt, clntToutK=clntToutK, clntTinK=clntTinK, mixedBatLoopToutK=mixedBatLoopToutK)


def pwrEstimation(s, uu, e, c, cmp):
    """Electrical power estimate (the NMPC cost). Not needed for dx/dt."""
    h2, h1 = cmp.compressor.h_out, cmp.compressor.h_in
    mf_compRef = cmp.compressor.refMf
    mf_blower = uu.cabAirMfIn
    density_glycol = 1100
    mf_bat = cmp.batPump.clntMf
    mf_clntMot = cmp.motPump.clntMf
    friction_coeff = (5e-6 * 64 / (2 * 0.0092 ** 2 * 1.5e-5)) * ((0.4 + 0.736) / 2)
    p_comElecDemand = mf_compRef * (h2 - h1) / 0.9 / 0.9
    blowerPwr = 11.6064 * mf_blower / 0.8
    delta_P_batpump = 6 * friction_coeff * mf_clntMot ** 2
    batpump_pwrEst = (mf_clntMot * delta_P_batpump) / density_glycol / 0.88
    k_heater = (2e-4 * 1e6) / 0.15 ** 2
    k_bat = (1e-5 * 1e6) / 0.02 ** 2
    p_DropAtMotPump = k_heater * mf_bat ** 2 + k_bat * mf_bat ** 2
    motPump_pwrEst = (mf_clntMot * p_DropAtMotPump) / density_glycol / 0.88
    radFan_pwrEst = 1701 * (uu.radFanrpm / 3000) ** 3 / (0.85 * 0.95)
    cmp.pwrEst = _S(pwrCompElec=p_comElecDemand, pwrBlowElec=blowerPwr,
                    pwrPumpMotElec=motPump_pwrEst, pwrPumpBatElec=batpump_pwrEst,
                    pwrTem=p_comElecDemand + blowerPwr + motPump_pwrEst + batpump_pwrEst + uu.heaterPwr + radFan_pwrEst)


# ═══════════════════════════════════════════════════════════════════ integration helpers
def rk4_step(x, u, est, ctrl, gamma, dt=1.0):
    """One RK4 step with inputs, parameters and γ held constant over the step (as in the NMPC)."""
    f = lambda xx: com_xdot(xx, u, est, ctrl, gamma)
    k1 = f(x); k2 = f(x + 0.5 * dt * k1); k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(x0, U, EST, CTRL, GAMMA, dt=1.0):
    """Simulate L steps. U [B, L, 6], EST/CTRL dicts of [B, L], GAMMA [B, L, 11] -> states [B, L, 9] (steps 1..L)."""
    x, out = x0, []
    for k in range(U.shape[1]):
        x = rk4_step(x, U[:, k], {n: v[:, k] for n, v in EST.items()},
                     {n: v[:, k] for n, v in CTRL.items()}, GAMMA[:, k], dt)
        out.append(x)
    return torch.stack(out, 1)
