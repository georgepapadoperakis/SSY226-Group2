% Export_data_to_Python.m
% Extracts 1 Hz ground-truth states X and inputs U directly from your Simscape runs:
% Results/run_2026_11_09_11_20_32/simlog_rule_T5.mat, T7.mat, T10.mat

clear; clc;

simlog_dir = 'Results/run_2026_11_09_11_20_32';
temps = [5, 7, 10];
t_grid = (0:1:3600)'; % Uniform 1 Hz grid (3601 steps)
N_steps = numel(t_grid);

out_struct = struct();

for idx = 1:numel(temps)
    T_abs = temps(idx);
    simlog_file = fullfile(simlog_dir, sprintf('simlog_rule_T%d.mat', T_abs));
    tuning_file = sprintf('Data/tuning_data_%d.mat', T_abs);
    
    fprintf('Loading %s ...\n', simlog_file);
    L = load(simlog_file, 'simlog');
    simlog = L.simlog;
    
    % Also load tuning_data to grab pre-evaluated CoolProp fluid tables (sim_params)
    S = load(tuning_file);
    r = S.resampled;
    
    %% ── 1. Extract 9 Ground-Truth States from simlog (in SI units: K, -, Pa) ──
    % 1: motT (K)
    motT = get_series_1hz(simlog, {'Motor.Motor.M.T', 'Motor.Motor.T'}, t_grid, 'K');
    % 2: invT (K)
    invT = get_series_1hz(simlog, {'Inverter.Inverter.M.T', 'Inverter.Inverter.T'}, t_grid, 'K');
    % 3: dcdcT (K)
    dcdcT = get_series_1hz(simlog, {'DCDC.DC_DC_Converter.M.T', 'DCDC.DCDC.T'}, t_grid, 'K');
    % 4: SOC (-)
    batSoc = double(reshape(r.batSocMeas(1:N_steps), [], 1));
    % 5: batT (K) - mean of 4 battery packs (exact match to RunThermalComparison.m)
    batT1 = get_series_1hz(simlog, {'Battery.Pack_1.Thermal_Model.Battery_Thermal_Mass.T', 'Battery.Pack_1.Thermal_Model.T'}, t_grid, 'K');
    batT2 = get_series_1hz(simlog, {'Battery.Pack_2.Thermal_Model.Battery_Thermal_Mass.T', 'Battery.Pack_2.Thermal_Model.T'}, t_grid, 'K');
    batT3 = get_series_1hz(simlog, {'Battery.Pack_3.Thermal_Model.Battery_Thermal_Mass.T', 'Battery.Pack_3.Thermal_Model.T'}, t_grid, 'K');
    batT4 = get_series_1hz(simlog, {'Battery.Pack_4.Thermal_Model.Battery_Thermal_Mass.T', 'Battery.Pack_4.Thermal_Model.T'}, t_grid, 'K');
    batT = 0.25 * (batT1 + batT2 + batT3 + batT4);
    % 6 & 7: Compressor inlet and outlet pressures (Pa)
    compPinPa  = get_series_1hz(simlog, {'Compressor.Compressor.A.p'}, t_grid, 'Pa');
    compPoutPa = get_series_1hz(simlog, {'Compressor.Compressor.B.p'}, t_grid, 'Pa');
    % 8: Cabin interior thermal mass temperature cabIntT (K)
    T_glass = get_series_1hz(simlog, {'Cabin.Cabin_Heat_Transfer.Glass_Thermal_Mass.M.T'}, t_grid, 'K');
    T_doors = get_series_1hz(simlog, {'Cabin.Cabin_Heat_Transfer.Doors_Thermal_Mass.M.T'}, t_grid, 'K');
    T_roof  = get_series_1hz(simlog, {'Cabin.Cabin_Heat_Transfer.Roof_Thermal_Mass.M.T'},  t_grid, 'K');
    cabIntT = (T_glass + T_doors + T_roof) / 3.0;
    % 9: Cabin air temperature cabAirT (K)
    cabAirT = get_series_1hz(simlog, {'Cabin.Cabin_Air_Volume.T_I'}, t_grid, 'K');
    
    X = [motT, invT, dcdcT, batSoc, batT, compPinPa, compPoutPa, cabIntT, cabAirT];
    
    % Also save the 23°C-setpoint Simscape reference from S.resampled for Figure 2 comparison
    cabInt_f2 = double(reshape(r.cabIntTInK(1:N_steps), [], 1));
    if mean(cabInt_f2) < 150, cabInt_f2 = cabInt_f2 + 273.15; end
    X_fig2 = [ ...
        double(reshape(r.motTmeasC(1:N_steps), [], 1)) + 273.15, ...
        double(reshape(r.invTmeasC(1:N_steps), [], 1)) + 273.15, ...
        double(reshape(r.dcdcTmeasC(1:N_steps), [], 1)) + 273.15, ...
        batSoc, ...
        double(reshape(r.batTmeasC(1:N_steps), [], 1)) + 273.15, ...
        double(reshape(r.compPinMpa(1:N_steps), [], 1)) * 1e6, ...
        double(reshape(r.compPoutMpa(1:N_steps), [], 1)) * 1e6, ...
        cabInt_f2, ...
        double(reshape(r.cabAirTmeasC(1:N_steps), [], 1)) + 273.15 ...
    ];
    
    %% ── 2. Extract 6 Control Inputs U from simlog (with Heater in Watts!) ──
    compNrpm = get_series_1hz(simlog, {'Compressor.Compressor.omega'}, t_grid, 'rpm');
    try
        cabAirMfIn = get_series_1hz(simlog, {'Cabin.Flow_Rate_Sensor_MA.M'}, t_grid, 'kg/s');
    catch
        V_blow = get_series_1hz(simlog, {'Blower.Blower.V'}, t_grid, 'm^3/s');
        cabAirMfIn = 1.225 * V_blow;
    end
    try
        motPumpNrpm = get_series_1hz(simlog, {'Motor_Pump.Motor_Pump.omega'}, t_grid, 'rpm');
    catch
        motPumpNrpm = double(reshape(r.motPumpNrpm(1:N_steps), [], 1));
    end
    try
        batPumpNrpm = get_series_1hz(simlog, {'Battery_Pump.Battery_Pump.omega'}, t_grid, 'rpm');
    catch
        batPumpNrpm = motPumpNrpm;
    end
    % Heater power directly in Watts from simlog (0 to 4000 W)
    heaterPwr = get_series_1hz(simlog, {'Heater.Controlled_Heat_Flow_Rate_Source.Q', ...
                                        'Heater.Controlled_Heat_Flow_Rate_Source.S'}, t_grid, 'W');
    radFanrpm = zeros(N_steps, 1);
    
    U = [compNrpm, cabAirMfIn, motPumpNrpm, batPumpNrpm, heaterPwr, radFanrpm];
    
    % U for the 23°C Figure 2 run (scaling heaterCmd [0,1] by 4000 W!)
    h_cmd = double(reshape(r.heaterCmd(1:N_steps), [], 1));
    if max(h_cmd) <= 10.0
        h_cmd = h_cmd * 4000.0;
    end
    U_fig2 = [ ...
        double(reshape(r.compNrpm(1:N_steps), [], 1)), ...
        double(reshape(r.cabAirMfIn(1:N_steps), [], 1)), ...
        double(reshape(r.motPumpNrpm(1:N_steps), [], 1)), ...
        double(reshape(r.motPumpNrpm(1:N_steps), [], 1)), ...
        h_cmd, ...
        zeros(N_steps, 1) ...
    ];
    
    %% ── 3. Unpack est and ctrl, and overwrite heat losses from simlog ──
    est  = unpack_struct_cell(S.sim_params, N_steps);
    ctrl = unpack_struct_cell(S.controlOutCell, N_steps);
    
    est.envT      = (-T_abs + 273.15) * ones(N_steps, 1); % Ensure Kelvin
    est.motQmeas  = get_series_1hz(simlog, {'Motor.Controlled_Heat_Flow_Rate_Source.S', 'Motor.Controlled_Heat_Flow_Rate_Source.Q'}, t_grid, 'W');
    est.invQmeas  = get_series_1hz(simlog, {'Inverter.Controlled_Heat_Flow_Rate_Source.S', 'Inverter.Controlled_Heat_Flow_Rate_Source.Q'}, t_grid, 'W');
    est.dcdcQmeas = get_series_1hz(simlog, {'DCDC.Controlled_Heat_Flow_Rate_Source.S', 'DCDC.Controlled_Heat_Flow_Rate_Source.Q'}, t_grid, 'W');
    est.batImeas  = double(reshape(r.batImeas(1:N_steps), [], 1));
    
    tag = sprintf('T%d', T_abs);
    out_struct.(tag).X       = X;
    out_struct.(tag).U       = U;
    out_struct.(tag).X_fig2  = X_fig2;
    out_struct.(tag).U_fig2  = U_fig2;
    out_struct.(tag).est     = est;
    out_struct.(tag).ctrl    = ctrl;
    out_struct.(tag).envTemp = -T_abs;
    out_struct.(tag).t       = t_grid;
end

Stuned = load('Data/tunedVariables_minus5_minus10.mat');
TV = Stuned.tunedVariables;
out_struct.baseline_TV.states     = double(TV.states);
out_struct.baseline_TV.parameters = double(TV.parameters);
out_struct.baseline_TV.ambient    = double(TV.ambient);

save('Data/python_training_data.mat', '-struct', 'out_struct', '-v7');
fprintf('✓ Successfully exported simlog_rule_Tx runs to Data/python_training_data.mat!\n');

%% ── Helper: traverse simlog node path and interpolate to 1 Hz ──
function y_1hz = get_series_1hz(simlog_root, path_candidates, t_grid, unit_str)
    for c = 1:numel(path_candidates)
        try
            parts = strsplit(path_candidates{c}, '.');
            node = simlog_root;
            for p = 1:numel(parts)
                node = node.(parts{p});
            end
            t_raw = node.series.time;
            if nargin >= 4 && ~isempty(unit_str)
                v_raw = node.series.values(unit_str);
            else
                v_raw = node.series.values;
            end
            y_1hz = double(interp1(t_raw, v_raw, t_grid, 'linear', 'extrap'));
            return;
        catch
        end
    end
    error('Could not find any candidate path in simlog: %s', strjoin(path_candidates, ', '));
end

function out = unpack_struct_cell(cell_arr, N_steps)
    first_valid = [];
    for k = 1:numel(cell_arr)
        if isstruct(cell_arr{k}), first_valid = cell_arr{k}; break; end
    end
    flds = fieldnames(first_valid);
    out = struct();
    for f = 1:numel(flds)
        fn = flds{f};
        vec = zeros(N_steps, 1);
        last_val = double(first_valid.(fn)(1));
        for k = 1:N_steps
            if k <= numel(cell_arr) && isstruct(cell_arr{k}) && isfield(cell_arr{k}, fn)
                v = cell_arr{k}.(fn);
                if ~isempty(v) && isfinite(double(v(1))), last_val = double(v(1)); end
            end
            vec(k) = last_val;
        end
        out.(fn) = vec;
    end
end