function file = export_training_data(simOut, simlog, T_amb_C, ctrl, outdir)
% EXPORT_TRAINING_DATA  Save one simulation run as 1-Hz training data.
%
%   file = export_training_data(simOut, simlog, T_amb_C, ctrl, outdir)
%
%   Writes  <outdir>/training_data_<ctrl>_T<|T|>.mat  containing
%     trainingData  struct with 1-Hz signals (time in s, temperatures in degC, powers in W)
%     missing    names of wanted signals that could not be found (fill in later)
%
%   Called automatically by RunThermalComparison after each run.

    t_sim = simlog.Motor.Motor.T.series.time;   % 'time' is a method: call it first, then index
    t_end = t_sim(end);
    t1hz  = (0:floor(t_end))';

    trainingData = struct();
    trainingData.t_s     = t1hz;
    trainingData.T_amb_C = T_amb_C;
    trainingData.ctrl    = ctrl;
    missing    = {};

    % ---- Signals with confirmed Simscape paths (same as compute_energy_results) ----
    [trainingData.motT_C,  missing] = grab(simlog, 'Motor.Motor.T',                      'degC', t1hz, missing);
    [trainingData.invT_C,  missing] = grab(simlog, 'Inverter.Inverter.T',                'degC', t1hz, missing);
    [trainingData.dcdcT_C, missing] = grab(simlog, 'DCDC.DC_DC_Converter.T',             'degC', t1hz, missing);
    [trainingData.cabAirT_C, missing] = grab(simlog, 'Cabin.Cabin_Air_Volume.T_I',       'degC', t1hz, missing);

    packs = zeros(numel(t1hz), 4);
    for k = 1:4
        [packs(:,k), missing] = grab(simlog, sprintf('Battery.Pack_%d.Thermal_Model.T', k), 'degC', t1hz, missing);
    end
    trainingData.batT_C = mean(packs, 2);

    [trainingData.compPwr_W,    missing] = grab(simlog, 'Compressor.Compressor.mechanical_power',        'W', t1hz, missing);
    [trainingData.motPumpPwr_W, missing] = grab(simlog, 'Motor_Pump.Motor_Pump.mechanical_power',        'W', t1hz, missing);
    [trainingData.batPumpPwr_W, missing] = grab(simlog, 'Battery_Pump.Battery_Pump.mechanical_power',    'W', t1hz, missing);
    [trainingData.blowerPwr_W,  missing] = grab(simlog, 'Blower.Blower.power',                           'W', t1hz, missing);
    [trainingData.fanPwr_W,     missing] = grab(simlog, 'Radiator.Fan.mechanical_power',                 'W', t1hz, missing);
    [trainingData.heaterQ_W,    missing] = grab(simlog, 'Heater.Controlled_Heat_Flow_Rate_Source.Q',     'W', t1hz, missing);

    % ---- Refrigerant loop ----
    [trainingData.compPin_MPa,  missing] = grab(simlog, 'Compressor.Compressor.A.p',   'MPa',  t1hz, missing);
    [trainingData.compPout_MPa, missing] = grab(simlog, 'Compressor.Compressor.B.p',   'MPa',  t1hz, missing);
    [trainingData.compNrpm,     missing] = grab(simlog, 'Compressor.Compressor.omega', 'rpm',  t1hz, missing);
    [trainingData.refMf_kgs,    missing] = grab(simlog, 'Compressor.Compressor.mdot_A','kg/s', t1hz, missing);

    % ---- Air side ----
    [trainingData.cabAirMfIn_kgs, missing] = grab(simlog, 'Blower.Blower.mdot_A',      'kg/s', t1hz, missing);
    [trainingData.radFanrpm,      missing] = grab(simlog, 'Radiator.Fan.omega',        'rpm',  t1hz, missing);

    % ---- Coolant loops ----
    [trainingData.motPumpNrpm,  missing] = grab(simlog, 'Motor_Pump.Motor_Pump.omega',      'rpm',  t1hz, missing);
    [trainingData.batPumpNrpm,  missing] = grab(simlog, 'Battery_Pump.Battery_Pump.omega',  'rpm',  t1hz, missing);
    [trainingData.motClntMf_kgs, missing] = grab(simlog, 'Motor_Pump.Motor_Pump.mdot_A',    'kg/s', t1hz, missing);
    [trainingData.batClntMf_kgs, missing] = grab(simlog, 'Battery_Pump.Battery_Pump.mdot_A','kg/s', t1hz, missing);
    trainingData.heaterCmd = trainingData.heaterQ_W / 4000;   % heater max power 4000 W

    % ---- Heat loads and battery ----
    [trainingData.motQgen_W,  missing] = grab(simlog, 'Motor.Controlled_Heat_Flow_Rate_Source.Q',    'W', t1hz, missing);
    [trainingData.invQgen_W,  missing] = grab(simlog, 'Inverter.Controlled_Heat_Flow_Rate_Source.Q', 'W', t1hz, missing);
    [trainingData.dcdcQgen_W, missing] = grab(simlog, 'DCDC.Controlled_Heat_Flow_Rate_Source.Q',     'W', t1hz, missing);
    [trainingData.batI_A,     missing] = grab(simlog, 'Battery.Current_Demand.i',                     'A', t1hz, missing);

    soc = zeros(numel(t1hz), 4);
    for k = 1:4
        [soc(:,k), missing] = grab(simlog, sprintf('Battery.Pack_%d.Lithium_Cell_1RC.Em_table.SOC', k), '', t1hz, missing);
    end
    trainingData.batSoc = mean(soc, 2);

    % ---- Cabin interior and coolant/vent temperatures (checked against old tuning_data) ----
    walls = {'Glass', 'Doors', 'Roof'};
    wallT = zeros(numel(t1hz), 3);
    for k = 1:3
        [wallT(:,k), missing] = grab(simlog, sprintf('Cabin.Cabin_Heat_Transfer.%s_Thermal_Mass.T', walls{k}), 'degC', t1hz, missing);
    end
    trainingData.cabIntT_C = mean(wallT, 2);                        % simple average (kept for compatibility)
    trainingData.glassT_C = wallT(:,1); trainingData.doorsT_C = wallT(:,2); trainingData.roofT_C = wallT(:,3);
    % Same definition as the COM's cabin-interior state (cabMdl in dynamics_mexfreeNew.m): weighted by m*cp
    wallC = [36.4*840, 75.0*500, 30.0*500];                         % glass, doors, roof [J/K]
    trainingData.cabIntTw_C = wallT * wallC(:) / sum(wallC);
    [trainingData.clntInvTin_C,     missing] = grab(simlog, 'Inverter.Coolant_Jacket.A.T',                        'degC', t1hz, missing);
    [trainingData.heaterClntTout_C, missing] = grab(simlog, 'Heater.Heater_Coolant_Volume.T_I',                   'degC', t1hz, missing);
    [trainingData.ventAirT_C,       missing] = grab(simlog, 'Inner_Condenser.Pressure_Temperature_Sensor_MA1.T',  'degC', t1hz, missing);

    % ---- Compressor inlet state -> superheat (computed in Python with CoolProp from p and u, or T if logged) ----
    [trainingData.compInU_kJkg, missing] = grab(simlog, 'Compressor.Compressor.A.u', 'kJ/kg', t1hz, missing);
    [trainingData.compInT_C,    missing] = grab(simlog, 'Compressor.Compressor.A.T', 'degC',  t1hz, missing);

    % ---- Vehicle speed: not in the Simscape log, taken from the drive-cycle file ----
    % (same file and variable as init_nmpc.m; the Scenario block uses the 'DriveCycle' entry, in km/h)
    try
        V  = load('ElectricVehicleThermalManagementWithHeatPumpVehicleSpeed.mat');
        ts = V.DriveCycle{1};
        [tv, iu] = unique(ts.Time(:));
        dv = ts.Data(:);
        trainingData.vehSpeed_kmh = interp1(tv, dv(iu), t1hz, 'linear', 0);
        trainingData.vehSpeed_ms  = trainingData.vehSpeed_kmh / 3.6;
    catch
        trainingData.vehSpeed_kmh = NaN(size(t1hz));
        trainingData.vehSpeed_ms  = NaN(size(t1hz));
        missing{end+1} = 'vehSpeed';
    end

    file = fullfile(outdir, sprintf('training_data_%s_T%d.mat', ctrl, abs(T_amb_C)));
    save(file, 'trainingData', 'missing');
    fprintf('[Export] %s  (%d s, %d signals still to map)\n', file, numel(t1hz), numel(missing));
end

% ---------------------------------------------------------------------------
function [y, missing] = grab(simlog, path, unit, t1hz, missing)
    try
        node = simlog;
        parts = strsplit(path, '.');
        for p = parts
            node = node.(p{1});
        end
        t = node.series.time;
        if isempty(unit), v = node.series.values(); else, v = node.series.values(unit); end
        [t, iu] = unique(t);                       % solver can repeat time points
        y = interp1(t, v(iu), t1hz, 'linear', 'extrap');
    catch
        y = NaN(size(t1hz));
        missing{end+1} = path;
    end
end
