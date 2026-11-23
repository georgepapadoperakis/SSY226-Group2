function out = export_heat_balance(simlogFile)
% EXPORT_HEAT_BALANCE  Heat flows of one Simscape run at 1 Hz (for checking the COM's energy paths).
%
%   export_heat_balance('..\Results\run_2026_11_15_13_52_12\simlog_rule_T12.mat')
%
% Writes heat_balance_<name>.mat next to the simlog with struct hb (all heat flows in W, temps in degC).
% Signs follow Simscape (Q into the block's first port is positive) - check per signal before using.
    S = load(simlogFile, 'simlog'); simlog = S.simlog;
    tt = simlog.Motor.Motor.T.series.time; t1 = (0:floor(tt(end)))';
    hb = struct('t_s', t1);
    sig = {
     'Q_motor_mass',        'Motor.Motor.Q',                                    'W'
     'Q_inverter_mass',     'Inverter.Inverter.Q',                              'W'
     'Q_dcdc_mass',         'DCDC.DC_DC_Converter.Q',                           'W'
     'Q_motor_jacket',      'Motor.Coolant_Jacket.segment.Q_H',                 'W'
     'Q_inverter_jacket',   'Inverter.Coolant_Jacket.segment.Q_H',              'W'
     'Q_dcdc_jacket',       'DCDC.Coolant_Jacket.segment.Q_H',                  'W'
     'Q_motor_gen',         'Motor.Controlled_Heat_Flow_Rate_Source.Q',         'W'
     'Q_inverter_gen',      'Inverter.Controlled_Heat_Flow_Rate_Source.Q',      'W'
     'Q_dcdc_gen',          'DCDC.Controlled_Heat_Flow_Rate_Source.Q',          'W'
     'Q_bat1_mass',         'Battery.Pack_1.Thermal_Model.Battery_Thermal_Mass.Q', 'W'
     'Q_bat1_gen',          'Battery.Pack_1.Thermal_Model.Controlled_Heat_Flow_Rate_Source.Q', 'W'
     'Q_htx',               'Heat_Exchanger.Heat_Exchanger.Q',                  'W'
     'Q_radiator',          'Radiator.Radiator.Q',                              'W'
     'eff_radiator',        'Radiator.Radiator.effectiveness',                  '1'
     'Q_condevap',          'Condenser_Evaporator.Condenser_Evaporator.Q',      'W'
     'Q_innercond',         'Inner_Condenser.Inner_Condenser.Q',                'W'
     'Q_chiller',           'Chiller.Chiller.Q',                                'W'
     'Q_evaporator',        'Evaporator.Evaporator.Q',                          'W'
     'Q_heater_src',        'Heater.Controlled_Heat_Flow_Rate_Source.Q',        'W'
     'Q_heater_conv',       'Heater.Convective_Heat_Transfer.Q',                'W'
     'Q_heater_battery',    'Heater.Battery_Heater.Q',                          'W'
     'Q_mottank_conv1',     'Motor_Pump.Heat_Transfer.Convective_Heat_Transfer.Q',   'W'
     'Q_mottank_conv2',     'Motor_Pump.Heat_Transfer.Convective_Heat_Transfer1.Q',  'W'
     'Q_mottank_env',       'Motor_Pump.Heat_Transfer.Environment_Temperature.Q',    'W'
     'Q_battank_conv1',     'Battery_Pump.Heat_Transfer.Convective_Heat_Transfer.Q', 'W'
     'Q_battank_conv2',     'Battery_Pump.Heat_Transfer.Convective_Heat_Transfer1.Q','W'
     'Q_battank_env',       'Battery_Pump.Heat_Transfer.Environment_Temperature.Q',  'W'
     'Q_mottank_H1',        'Motor_Pump.Motor_Coolant_Tank.Q_H1',               'W'
     'Q_mottank_H2',        'Motor_Pump.Motor_Coolant_Tank.Q_H2',               'W'
     'Q_battank_H1',        'Battery_Pump.Battery_Coolant_Tank.Q_H1',           'W'
     'Q_battank_H2',        'Battery_Pump.Battery_Coolant_Tank.Q_H2',           'W'
     'T_mottank_liquid',    'Motor_Pump.Motor_Coolant_Tank.T_liquid',           'degC'
     'T_battank_liquid',    'Battery_Pump.Battery_Coolant_Tank.T_liquid',       'degC'
     'm_mottank_liquid',    'Motor_Pump.Motor_Coolant_Tank.mass_liquid',        'kg'
     'm_battank_liquid',    'Battery_Pump.Battery_Coolant_Tank.mass_liquid',    'kg'
    };
    missing = {};
    for i = 1:size(sig, 1)
        try
            node = simlog;
            for p = strsplit(sig{i, 2}, '.'), node = node.(p{1}); end
            t = node.series.time;
            if strcmp(sig{i, 3}, '1'), v = node.series.values(); else, v = node.series.values(sig{i, 3}); end
            [t, iu] = unique(t);
            hb.(sig{i, 1}) = interp1(t, v(iu), t1, 'linear', 'extrap');
        catch
            hb.(sig{i, 1}) = NaN(size(t1)); missing{end+1} = sig{i, 2}; %#ok<AGROW>
        end
    end
    [folder, name] = fileparts(simlogFile);
    out = fullfile(folder, ['heat_balance_' strrep(name, 'simlog_', '') '.mat']);
    save(out, 'hb', 'missing');
    fprintf('[heat balance] %s  (%d signals, %d missing)\n', out, size(sig, 1), numel(missing));
    if ~isempty(missing), disp(missing'); end
end
