# com_physics.py
import math
import torch
import torch.nn as nn

class COMPhysics(nn.Module):
    """
    Exact 1-to-1 differentiable PyTorch translation of dynamics_mexfreeNew.m.
    State vector x (B, 9):
        0: motT (K), 1: invT (K), 2: dcdcT (K), 3: batSoc (-), 4: batT (K),
        5: compPinPa (Pa), 6: compPoutPa (Pa), 7: cabIntT (K), 8: cabAirT (K)
    Control vector u (B, 6):
        0: compNrpm, 1: cabAirMfIn, 2: motPumpNrpm, 3: batPumpNrpm, 4: heaterPwr, 5: radFanrpm
    Parameter vector P (B, 11):
        P[:, 0]..P[:, 10] corresponding to P(1)..P(11) in dynamics_mexfreeNew.m
    """
    def __init__(self):
        super().__init__()
        self.NTU_MAX = 10.0

    def forward(self, x, u, est, ctrl, P):
        """
        Computes xdot = f(x, u, est, ctrl, P) for batch size B.
        est: dict of (B,) tensors from estimateParameters.m
        ctrl: dict of (B,) tensors from thermalLogicAndEstimation.m
        P: (B, 11) tensor of scaling parameters predicted by the Neural Network
        """
        # --- Unpack states (build_runtime_context) ---
        motT       = x[:, 0]
        invT       = x[:, 1]
        dcdcT      = x[:, 2]
        batSoc     = x[:, 3]
        batT       = x[:, 4]
        compPinPa  = x[:, 5]
        compPoutPa = x[:, 6]
        cabIntT    = x[:, 7]
        cabAirT    = x[:, 8]

        # --- Unpack inputs (with exact offsets from build_runtime_context) ---
        compNrpm    = u[:, 0]
        cabAirMfIn  = u[:, 1] + 1e-3
        motPumpNrpm = u[:, 2]
        batPumpNrpm = u[:, 3]
        heaterPwr   = u[:, 4]
        radFanrpm   = u[:, 5]
        batImeas    = est["batImeas"] + 1e-3

        # =====================================================================
        # 1. pumpMdl
        # =====================================================================
        pumpVolEff = 0.92
        pumpDispM3perRev = 0.02 * 1e-3
        densityCorrection = 1.0660
        clntRho_pump = 1100.0

        batClntMf = clntRho_pump * densityCorrection * (pumpDispM3perRev * (batPumpNrpm / 60.0)) * pumpVolEff + 1e-4
        motClntMf = clntRho_pump * densityCorrection * (pumpDispM3perRev * (motPumpNrpm / 60.0)) * pumpVolEff + 1e-4

        # =====================================================================
        # 2. compressorMdl
        # =====================================================================
        compDispM3perRev = 80e-6
        isenEff = 0.65
        mechEff = 0.9
        massFlowCorrection = 0.12

        compNrpm_safe = compNrpm + 1e-5
        h_in = est["compHin"]
        h_outS = est["compHoutS"]
        vNominalIn = est["compVnominalIn"]

        p_in_floor = 0.3e5
        eps_pFloor = 1e6
        compPinSafe = 0.5 * (compPinPa + p_in_floor + torch.sqrt((compPinPa - p_in_floor)**2 + eps_pFloor))

        pressureRatio = compPoutPa / compPinSafe
        volEff_calc = -0.0286 * pressureRatio + 1.0286
        k_smooth = 50.0
        # Softplus with numerical stability
        volEff = 0.01 + (1.0 / k_smooth) * torch.nn.functional.softplus(k_smooth * (volEff_calc - 0.01))

        angularVel = compNrpm_safe * (2.0 * math.pi / 60.0)
        refMf = volEff * angularVel * compDispM3perRev * massFlowCorrection / vNominalIn + 1e-4

        deltaH_isentropic = h_outS - h_in
        h_comp_out = h_in + deltaH_isentropic / isenEff

        temp_rise = 2.0 * ((h_outS - h_in) / (est["refCpVapHighP"] + 1e-6))
        compToutK = est["refTsatHighPK"] + temp_rise

        # =====================================================================
        # 3. innerCondenserMdl
        # =====================================================================
        refArea_ic = 7.854e-05
        airArea_ic = 0.04
        totalHxArea_ic = 1.1088 + 0.9952
        wallThermalResistance_ic = 4.1868e-07
        ntuAreaFactor_ic = 18.0 * 40.0
        refD_ic = 0.0100
        airD_ic = 0.0190

        refV_ic = refMf / (est["innerCondRefRho"] * refArea_ic + 1e-6)
        airV_ic = (ctrl["cmdBlendAir"] * cabAirMfIn / (est["cabAirRho"] * airArea_ic + 1e-6)) + 1e-6

        refRe_ic = (est["innerCondRefRho"] * refV_ic * refD_ic) / (est["innerCondRefMu"] + 1e-6)
        airRe_ic = (est["cabAirRho"] * airV_ic * airD_ic) / (est["cabAirMu"] + 1e-6)
        refPr_ic = (est["innerCondRefCp"] * est["innerCondRefMu"]) / (est["innerCondRefK"] + 1e-6)
        airPr_ic = (est["cabAirCp"] * est["cabAirMu"]) / (est["cabAirK"] + 1e-6)

        refNu_ic = 0.023 * (refRe_ic**0.8) * (refPr_ic**0.33)
        airNu_ic = 0.023 * (airRe_ic**0.8) * (airPr_ic**0.33)
        refU_ic = (refNu_ic * est["innerCondRefK"]) / (refD_ic + 1e-6)
        airU_ic = (airNu_ic * est["cabAirK"]) / (airD_ic + 1e-6)

        thermalResistance_ic = (1.0 / (refU_ic * refArea_ic + 1e-6)) + (1.0 / (airU_ic * airArea_ic + 1e-6)) + wallThermalResistance_ic
        c_air_ic = cabAirMfIn * est["cabAirCp"]
        c_min_ic = c_air_ic

        z_air = 5e4 * (cabAirMfIn - 1e-4)
        sigmoid_gate_c_air = 0.5 * (1.0 + torch.tanh(z_air))

        NTU_raw_ic = ((ntuAreaFactor_ic * totalHxArea_ic) / (thermalResistance_ic * c_min_ic + 1e-6)) * sigmoid_gate_c_air
        NTU_ic = self.NTU_MAX * torch.tanh(NTU_raw_ic / self.NTU_MAX)
        eff_ic = 1.0 - torch.exp(-NTU_ic)

        qMax_ic = c_min_ic * (compToutK - cabAirT)
        qActual_ic = eff_ic * qMax_ic

        delta_T_air_ic = (qActual_ic / (c_air_ic + 1e-6)) * sigmoid_gate_c_air
        cabAirToutK_ic = cabAirT + delta_T_air_ic

        c_ref_ic = refMf * est["innerCondRefCp"]
        z_ref = 5e4 * (refMf - 1e-4)
        sigmoid_gate_c_ref = 0.5 * (1.0 + torch.tanh(z_ref))

        delta_T_ref_ic = (0.35 * qActual_ic / (c_ref_ic + 1e-6)) * sigmoid_gate_c_ref
        refToutK_ic = compToutK - delta_T_ref_ic
        h_out_ic = est["refHsatLiqHighP"] + est["innerCondRefCp"] * (refToutK_ic - est["refTsatLiqHighPK"])

        # =====================================================================
        # 4. hpExvMdl & 5. condEvapMdl
        # =====================================================================
        cmdExvHp = ctrl["cmdExvHeatpump"]
        h_subcooled_liq = est["refHsatLiqHighP"] + est["refCpLiqHighP"] * (refToutK_ic - est["refTsatLiqHighPK"])
        refToutK_hpExv = est["refTsatLiqLowPK"] * cmdExvHp + refToutK_ic * (1.0 - cmdExvHp)

        # In condEvapMdl, airMf is explicitly set to 0 in dynamics_mexfreeNew.m:
        # airMf = 0 -> c_air = 0 -> qMax = 0 -> qActual_ce = 0, refToutK_ce = refToutK_hpExv
        refToutK_ce = refToutK_hpExv
        qActual_ce = torch.zeros_like(refMf)

        # =====================================================================
        # 6. solve_loop_htx_in_explicit (Analytical Coolant Loop Solver)
        # =====================================================================
        clntCp = est["clntCp"]
        clntRho = est["clntRho"]
        clntK = est["clntK"]
        clntMu = est["clntMu"]

        m_mot = motClntMf
        m_tot_bat = batClntMf
        split = 0.11
        m_dcdc = split * m_tot_bat
        m_bat = m_tot_bat - m_dcdc

        # local_htx_affine
        refArea_hx = 7.854e-05
        clntArea_hx = 0.00028353
        totalHxArea_hx = 64.0 * math.pi * 0.0035 * 0.1
        refD_hx = math.sqrt((4.0 * refArea_hx) / math.pi)
        clntD_hx = math.sqrt((4.0 * clntArea_hx) / math.pi)

        refV_hx = refMf / (est["condEvapRefRho"] * refArea_hx + 1e-6)
        clntV_hx = m_mot / (clntRho * clntArea_hx + 1e-6)
        refRe_hx = (est["condEvapRefRho"] * refV_hx * refD_hx) / (est["condEvapRefMu"] + 1e-6)
        clntRe_hx = (clntRho * clntV_hx * clntD_hx) / (clntMu + 1e-6)
        refPr_hx = (est["condEvapRefCp"] * est["condEvapRefMu"]) / (est["condEvapRefK"] + 1e-6)
        clntPr_hx = (clntCp * clntMu) / (clntK + 1e-6)
        refNu_hx = 0.023 * (refRe_hx**0.8) * (refPr_hx**0.33)
        clntNu_hx = 0.023 * (clntRe_hx**0.8) * (clntPr_hx**0.33)
        h_ref_hx = (refNu_hx * est["condEvapRefK"]) / (refD_hx + 1e-6)
        h_clnt_hx = (clntNu_hx * clntK) / (clntD_hx + 1e-6)

        thermRes_hx = (1.0 / (h_ref_hx * totalHxArea_hx + 1e-6)) + (1.0 / (h_clnt_hx * totalHxArea_hx + 1e-6))
        UA_hx = 1.0 / (thermRes_hx + 1e-6)
        c_clnt_mot = m_mot * clntCp
        NTU_hx = self.NTU_MAX * torch.tanh((0.9 * UA_hx / (c_clnt_mot + 1e-6)) / self.NTU_MAX)
        A_htx = 1.0 - torch.exp(-NTU_hx)
        a_htx = 1.0 - A_htx
        b_htx = A_htx * refToutK_ce

        # local_heater_affine
        dT_heat = (0.85 * heaterPwr) / (1.5 * m_tot_bat * clntCp + 1e-6)
        a_heat = 1.0
        b_heat = dT_heat

        a_T2_T0 = a_heat * a_htx
        b_T2 = a_heat * b_htx + b_heat

        # local_branch_affine_bat, dcdc, mot, inv
        clntChannelD = 0.0092
        hxAreaFactor = 0.736

        def branch_affine(m_flow, K_fluid, conv_factor, cond_factor, ntu_scale, T_comp):
            h_c = 3.66 * K_fluid / clntChannelD
            UA_c = h_c * hxAreaFactor * conv_factor
            c_c = m_flow * clntCp
            ntu = self.NTU_MAX * torch.tanh((ntu_scale * UA_c / (c_c + 1e-6)) / self.NTU_MAX)
            eff = 1.0 - torch.exp(-ntu)
            k_cond = cond_factor * hxAreaFactor / clntChannelD
            k_cond_eff = self.NTU_MAX * torch.tanh((k_cond / (c_c + 1e-6)) / self.NTU_MAX)
            S = eff + k_cond_eff
            return 1.0 - S, S * T_comp

        a_bat, b_bat   = branch_affine(m_bat,  clntK, 200.0, 0.7,   0.1, batT)
        a_dcdc, b_dcdc = branch_affine(m_dcdc, 0.254, 1.0,   0.245, 1.0, dcdcT)
        a_mot, b_mot   = branch_affine(m_mot,  clntK, 6.0,   0.254, 1.0, motT)
        a_inv, b_inv   = branch_affine(m_mot,  clntK, 6.0,   0.254, 0.1, invT)

        w_bat = m_bat / (m_bat + m_dcdc + 1e-12)
        w_dcdc = 1.0 - w_bat
        a_mix = w_bat * a_bat + w_dcdc * a_dcdc
        b_mix = w_bat * b_bat + w_dcdc * b_dcdc

        cmdPS = ctrl["cmdParallelSerial"]
        a_Tmotin_T0 = cmdPS * (a_mix * a_T2_T0) + (1.0 - cmdPS) * a_htx
        b_Tmotin = cmdPS * (a_mix * b_T2 + b_mix) + (1.0 - cmdPS) * b_htx

        a_chain = a_mot * a_Tmotin_T0
        b_chain = a_mot * b_Tmotin + b_mot
        alpha = a_inv * a_chain
        beta = a_inv * b_chain + b_inv
        htxClntTinK = beta / (1.0 - alpha + 1e-6)

        # =====================================================================
        # 7. htxMdl, chillerMdl, heaterMdl
        # =====================================================================
        qMax_hx = c_clnt_mot * (htxClntTinK - refToutK_ce)
        qActual_hx = A_htx * qMax_hx
        htxClntToutK = htxClntTinK - qActual_hx / (c_clnt_mot + 1e-6)

        # chillerMdl sets clntToutK = clntTinK = htxClntToutK in cold climate
        chillerClntToutK = htxClntToutK
        heaterClntToutK = chillerClntToutK + dT_heat

        # =====================================================================
        # 8. batMdl (uses P[:, 6] = P(7) and P[:, 2] = P(3))
        # =====================================================================
        batMassKg = 2.5 * 20.0 * 4.0
        batCp = 795.0
        clntTinK_bat = heaterClntToutK
        c_clnt_bat = m_bat * clntCp

        h_conv_bat = 3.66 * clntK / clntChannelD
        UA_bat = h_conv_bat * hxAreaFactor * 200.0
        NTU_raw_bat = (P[:, 6] * UA_bat) / (c_clnt_bat + 1e-6)
        NTU_bat = self.NTU_MAX * torch.tanh(NTU_raw_bat / self.NTU_MAX)
        eff_bat = 1.0 - torch.exp(-NTU_bat)

        qConv_bat = eff_bat * c_clnt_bat * (batT - clntTinK_bat)
        qCond_bat = (0.7 * hxAreaFactor / clntChannelD) * (batT - clntTinK_bat)
        qTotal_bat = qConv_bat + qCond_bat

        capacityAh = 0.0075 * (batT**2) - 4.5401 * batT + 962.67
        p00, p10, p01 = 0.5345, -0.00348, -0.02749
        p20, p11, p02 = 5.748e-06, 7.589e-05, 0.02773
        p21, p12, p03 = 9.909e-08, -0.0001321, 0.008301
        res_poly = (p00 + p10*batT + p01*batSoc + p20*(batT**2) + p11*batT*batSoc + p02*(batSoc**2)
                    + p21*(batT**2)*batSoc + p12*batT*(batSoc**2) + p03*(batSoc**3))
        resistanceOhm = 0.5 * (res_poly + 1e-8 + torch.sqrt((res_poly - 1e-8)**2 + 1e-18))

        batSoc_dt = -batImeas / (capacityAh * 4.0 * (44.0 * 20.0) + 1e-6)
        qJoule = 70.0 * (batImeas**2) * resistanceOhm
        thermalMass_bat = 0.75 * batMassKg * batCp
        batT_dt = P[:, 2] * (0.98 * qJoule - qTotal_bat) / (thermalMass_bat + 1e-6)
        batClntToutK = clntTinK_bat + qTotal_bat / (c_clnt_bat + 1e-6)

        # =====================================================================
        # 9. dcdcMdl
        # =====================================================================
        c_clnt_dcdc = m_dcdc * clntCp
        C0_dcdc = 1e-4 * clntCp
        gate_dcdc = torch.sigmoid(500.0 * (c_clnt_dcdc - C0_dcdc))
        UA_dcdc = (3.66 * 0.254 / clntChannelD) * hxAreaFactor * 1.0
        NTU_dcdc = self.NTU_MAX * torch.tanh((UA_dcdc / (c_clnt_dcdc + 1e-6) * gate_dcdc) / self.NTU_MAX)
        eff_dcdc = 1.0 - torch.exp(-NTU_dcdc)

        qConv_dcdc = eff_dcdc * c_clnt_dcdc * (dcdcT - clntTinK_bat)
        qCond_dcdc = (0.245 * hxAreaFactor / clntChannelD) * (dcdcT - clntTinK_bat)
        qRemoved_dcdc = qConv_dcdc + qCond_dcdc
        dcdcT_dt = 0.05 * (est["dcdcQmeas"] - qRemoved_dcdc) / (0.1 * 1.0 * 447.0 + 1e-6)
        dcdcClntToutK = clntTinK_bat + (qRemoved_dcdc / (c_clnt_dcdc + 1e-6)) * gate_dcdc

        # =====================================================================
        # 10. motMdl (uses P[:, 4] = P(5))
        # =====================================================================
        mixedBatLoopToutK = (batClntToutK * m_bat + dcdcClntToutK * m_dcdc) / (m_bat + m_dcdc + 1e-12)
        clntTinK_mot = cmdPS * mixedBatLoopToutK + (1.0 - cmdPS) * htxClntToutK

        UA_mot = (3.66 * clntK / clntChannelD) * hxAreaFactor * 6.0
        gate_mot = 0.5 * (1.0 + torch.tanh(1000.0 * (m_mot - 1e-3)))
        NTU_mot = self.NTU_MAX * torch.tanh(((UA_mot / (c_clnt_mot + 1e-6)) * gate_mot) / self.NTU_MAX)
        eff_mot = 1.0 - torch.exp(-NTU_mot)

        qConv_mot = eff_mot * c_clnt_mot * (motT - clntTinK_mot)
        qCond_mot = (0.254 * hxAreaFactor / clntChannelD) * (motT - clntTinK_mot)
        qRemoved_mot = qConv_mot + qCond_mot
        motT_dt = P[:, 4] * (est["motQmeas"] - qRemoved_mot) / (46.0 * 447.0 + 1e-6)
        motClntToutK = clntTinK_mot + (qRemoved_mot / (c_clnt_mot + 1e-6)) * gate_mot

        # =====================================================================
        # 11. invMdl (uses P[:, 3] = P(4))
        # =====================================================================
        clntTinK_inv = motClntToutK
        UA_inv = UA_mot
        NTU_inv = self.NTU_MAX * torch.tanh((0.1 * UA_inv / (c_clnt_mot + 1e-6)) / self.NTU_MAX)
        eff_inv = 1.0 - torch.exp(-NTU_inv)

        qConv_inv = eff_inv * c_clnt_mot * (invT - clntTinK_inv)
        qCond_inv = (0.254 * hxAreaFactor / clntChannelD) * (invT - clntTinK_inv)
        qRemoved_inv = qConv_inv + qCond_inv
        invT_dt = P[:, 3] * (est["invQmeas"] - qRemoved_inv) / (1.0 * 447.0 + 1e-6)

        # =====================================================================
        # 12. cabMdl (uses P[:, 0] = P(1), P[:, 1] = P(2), P[:, 7] = P(8))
        # =====================================================================
        R_glass = 0.01217 * (1.0 / (40.0 * 3.5) + 0.002 / (0.96 * 3.5))
        R_doors = 0.15076 * (1.0 / (40.0 * 4.0) + 0.03 / (0.08 * 4.0))
        R_roof  = 0.63668 * (1.0 / (40.0 * 2.0) + 0.02 / (0.08 * 2.0))
        R_total_env2int = R_glass + R_doors + R_roof
        thermalMass_int = 36.4 * 840.0 + 75.0 * 500.0 + 30.0 * 500.0

        q_env2int = (est["envT"] - cabIntT) / (R_total_env2int + 1e-6)
        q_air2int = (cabAirT - cabIntT) / (R_total_env2int + 1e-6)
        cabIntT_dt = (P[:, 0] * 42237.0 * (q_env2int + q_air2int)) / (2287.85 * thermalMass_int + 1e-6)

        thermalMass_air = 3.0 * 1.225 * 1006.0
        qHvac = cabAirMfIn * 1006.0 * (cabAirToutK_ic - cabAirT)
        R_int2air = 0.02475 * R_total_env2int
        q_int2air = (cabIntT - cabAirT) / (R_int2air + 1e-6)
        cabAirT_dt = (P[:, 1] * 0.01611 / (thermalMass_air + 1e-6)) * (qHvac + 70.0 + P[:, 7] * q_int2air)

        # =====================================================================
        # 13. pressureStateCalcMdl (uses P[:, 5]=P(6), P[:, 8]=P(9), P[:, 9]=P(10), P[:, 10]=P(11))
        # =====================================================================
        q_evap = qActual_hx + qActual_ce
        q_cond = qActual_ic
        h1, h2, h3, h4 = h_in, h_comp_out, h_out_ic, h_out_ic

        ye, yc = 0.8, 0.5
        Ve, Vc = 0.001, 0.001
        Mwe_ce, Mwc_cc = 5000.0, 5000.0

        f1 = P[:, 8] * q_evap + refMf * (h4 - h1)
        D11 = Ve * ((1.0 - ye) * est["d_rhoh_l_dp1"] + ye * est["d_rhoh_g_dp1"] - 1.0 + (Mwe_ce / Ve) * est["dTesat_dp1"])

        f2 = -q_cond + P[:, 5] * refMf * (h2 - h3)
        D22 = Vc * ((1.0 - yc) * est["d_rhoh_l_dp2"] + yc * est["d_rhoh_g_dp2"] - 1.0 + (Mwc_cc / Vc) * est["dTcsat_dp2"])

        sig_D11 = torch.sigmoid(100.0 * (torch.abs(D11) - 1e-4))
        sig_D22 = torch.sigmoid(100.0 * (torch.abs(D22) - 1e-4))

        compPin_dt  = (P[:, 10] * f1 / (500.0 * D11 + 1e-6)) * sig_D11
        compPout_dt = (P[:, 9]  * f2 / (1.0   * D22 + 1e-6)) * sig_D22

        return torch.stack([
            motT_dt, invT_dt, dcdcT_dt, batSoc_dt, batT_dt,
            compPin_dt, compPout_dt, cabIntT_dt, cabAirT_dt
        ], dim=-1)

    def step_rk4(self, x, u, est, ctrl, P, dt=1.0):
        """1-step RK4 integration (dt = 1.0 s) holding u, est, ctrl, and P constant over the step."""
        k1 = self.forward(x, u, est, ctrl, P)
        k2 = self.forward(x + 0.5 * dt * k1, u, est, ctrl, P)
        k3 = self.forward(x + 0.5 * dt * k2, u, est, ctrl, P)
        k4 = self.forward(x + dt * k3, u, est, ctrl, P)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)