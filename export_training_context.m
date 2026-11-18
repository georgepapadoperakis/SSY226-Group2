function export_training_context()
% EXPORT_TRAINING_CONTEXT: komplettera den befintliga numeriska exporten.
% Placera filen i huvudmappen SSY226-Group2 och kör export_training_context.
% Ingen ny simulering startas. De gamla exportfilerna ändras inte.
%
% Varje signal behåller sina egna tider. Vi bestämmer ännu inte:
% - om SOC/batteritemperatur ska representeras av ett pack eller ett medel,
% - hur glas/dörrar/tak ska bilda COM:s enda inredningstemperatur,
% - om ett kommando eller ett faktiskt värde motsvarar respektive COM-insignal.
% Därför exporteras relevanta kandidater separat och mappningen markeras ogranskad.

projektmapp = fileparts(mfilename('fullpath'));
korning = 'run_2026_11_11_10_22_21';
indata = fullfile(projektmapp, 'Results', korning);
utdata = fullfile(projektmapp, 'Data', 'NN_training', 'context_raw', korning);
assert(isfolder(indata), 'Hittar inte %s', indata);
if ~isfolder(utdata), mkdir(utdata); end

% Kolumner: exportnamn, exakt nodväg, önskad fysisk enhet.
% Alla vägar har kontrollerats mot er uppladdade signal_catalogue.csv.
% values(...,Units=...) gör enhetsomvandlingen, t.ex. MPa -> Pa och rad/s -> rpm.
spec = {
    'T_bat_pack1_K', 'Battery.Pack_1.Thermal_Model.T', 'K';
    'SOC_pack1', 'Battery.Pack_1.Lithium_Cell_1RC.Em_table.SOC', '1';
    'I_cell_pack1_A', 'Battery.Pack_1.Lithium_Cell_1RC.Em_table.i', 'A';
    'T_bat_pack2_K', 'Battery.Pack_2.Thermal_Model.T', 'K';
    'SOC_pack2', 'Battery.Pack_2.Lithium_Cell_1RC.Em_table.SOC', '1';
    'I_cell_pack2_A', 'Battery.Pack_2.Lithium_Cell_1RC.Em_table.i', 'A';
    'T_bat_pack3_K', 'Battery.Pack_3.Thermal_Model.T', 'K';
    'SOC_pack3', 'Battery.Pack_3.Lithium_Cell_1RC.Em_table.SOC', '1';
    'I_cell_pack3_A', 'Battery.Pack_3.Lithium_Cell_1RC.Em_table.i', 'A';
    'T_bat_pack4_K', 'Battery.Pack_4.Thermal_Model.T', 'K';
    'SOC_pack4', 'Battery.Pack_4.Lithium_Cell_1RC.Em_table.SOC', '1';
    'I_cell_pack4_A', 'Battery.Pack_4.Lithium_Cell_1RC.Em_table.i', 'A';
    'T_motor_K', 'Motor.Motor.T', 'K';
    'T_inverter_K', 'Inverter.Inverter.T', 'K';
    'T_dcdc_K', 'DCDC.DC_DC_Converter.T', 'K';
    'T_cabin_air_K', 'Cabin.Cabin_Air_Volume.T_I', 'K';
    'T_glass_K', 'Cabin.Cabin_Heat_Transfer.Glass_Thermal_Mass.T', 'K';
    'T_doors_K', 'Cabin.Cabin_Heat_Transfer.Doors_Thermal_Mass.T', 'K';
    'T_roof_K', 'Cabin.Cabin_Heat_Transfer.Roof_Thermal_Mass.T', 'K';
    'p_comp_A_Pa', 'Compressor.Compressor.A.p', 'Pa';
    'p_comp_B_Pa', 'Compressor.Compressor.B.p', 'Pa';
    'I_demand_command_A', 'Battery.Current_Demand.iT', 'A';
    'I_demand_physical_A', 'Battery.Current_Demand.i', 'A';
    'compressor_actual_rpm', 'Compressor.Compressor.omega', 'rpm';
    'motor_pump_actual_rpm', 'Motor_Pump.Motor_Pump.omega', 'rpm';
    'battery_pump_actual_rpm', 'Battery_Pump.Battery_Pump.omega', 'rpm';
    'fan_actual_rpm', 'Radiator.Fan.omega', 'rpm';
    'compressor_source_rpm', 'Compressor.Ideal_Angular_Velocity_Source.S', 'rpm';
    'motor_pump_source_rpm', 'Motor_Pump.Ideal_Angular_Velocity_Source.S', 'rpm';
    'battery_pump_source_rpm', 'Battery_Pump.Ideal_Angular_Velocity_Source.S', 'rpm';
    'fan_source_rpm', 'Radiator.Ideal_Angular_Velocity_Source.S', 'rpm';
    'blower_sensor_flow_kg_s', 'Blower.Flow_Rate_Sensor_MA.M', 'kg/s';
    'blower_mdot_A_kg_s', 'Blower.Blower.mdot_A', 'kg/s';
    'blower_mdot_B_kg_s', 'Blower.Blower.mdot_B', 'kg/s';
    'heater_source_command_W', 'Heater.Controlled_Heat_Flow_Rate_Source.S', 'W';
    'heater_heat_flow_W', 'Heater.Controlled_Heat_Flow_Rate_Source.Q', 'W';
    'Q_motor_source_W', 'Motor.Controlled_Heat_Flow_Rate_Source.S', 'W';
    'Q_inverter_source_W', 'Inverter.Controlled_Heat_Flow_Rate_Source.S', 'W';
    'Q_dcdc_source_W', 'DCDC.Controlled_Heat_Flow_Rate_Source.S', 'W';
    'Q_motor_mass_net_W', 'Motor.Motor.Q', 'W';
    'Q_inverter_mass_net_W', 'Inverter.Inverter.Q', 'W';
    'Q_dcdc_mass_net_W', 'DCDC.DC_DC_Converter.Q', 'W';
    'clnt_inverter_A_K', 'Inverter.SC7.Pressure_Temperature_Sensor_TL.A.T', 'K';
    'clnt_battery_SC3_K', 'Battery.SC3.Pressure_Temperature_Sensor_TL.T', 'K';
    'clnt_battery_SC4_K', 'Battery.SC4.Pressure_Temperature_Sensor_TL.T', 'K';
    'evap_valve_S_m', 'Evaporator_Expansion_Valve.Evaporator_Expansion_Valve.S', 'm';
    'hp_valve_S_m', 'Heat_Pump_Expansion_Valve.Heat_Pump_Expansion_Valve.S', 'm';
    'chiller_valve_S_m', 'Chiller_Expansion_Valve.Chiller_Expansion_Valve.S', 'm';
    'serial_valve1_S_m', 'Parallel_Serial_Mode_Valve_1.Parallel_Serial_Mode_Valve.S', 'm';
    'serial_valve2_S_m', 'Parallel_Serial_Mode_Valve_2.Parallel_Serial_Mode_Valve.S', 'm';
    'blend_AB_area_m2', 'Blend_Door.Blend_Door.Restriction_AB.AR', 'm^2';
    'blend_AC_area_m2', 'Blend_Door.Blend_Door.Restriction_AC.AR', 'm^2';
    'evap_open_fraction', 'Evaporator_Expansion_Valve.Evaporator_Expansion_Valve.opening_area_fraction', '1';
    'hp_open_fraction', 'Heat_Pump_Expansion_Valve.Heat_Pump_Expansion_Valve.opening_area_fraction', '1';
    'chiller_open_fraction', 'Chiller_Expansion_Valve.Chiller_Expansion_Valve.opening_area_fraction', '1';
};

for T_amb_C = [-5, -7, -10]
    kallfil = fullfile(indata, sprintf('simlog_rule_T%d.mat', abs(T_amb_C)));
    inlast = load(kallfil, 'simlog');
    assert(isfield(inlast, 'simlog'), 'simlog saknas i %s', kallfil);
    context = struct('schema_version', 1, 'T_amb_C', T_amb_C, ...
        'controller', 'rule', 'source_run', korning, ...
        'reviewed_mapping', false, 'signals', struct());
    fprintf('\nReading: %s\n', kallfil);
    for j = 1:size(spec, 1)
        namn = spec{j, 1};
        delar = strsplit(spec{j, 2}, '.');
        nod = inlast.simlog;
        for k = 1:numel(delar)
            nod = nod.(delar{k});
        end
        serie = nod.series;
        t = double(time(serie));
        y = double(values(serie, 'Units', spec{j, 3}));
        assert(isvector(t) && isvector(y), 'Inte en skalarserie: %s', spec{j,2});
        t = t(:); y = y(:);
        assert(~isempty(t) && numel(t)==numel(y), 'Fel storlek: %s', namn);
        assert(all(isfinite(t)) && all(isfinite(y)), 'NaN/Inf: %s', namn);
        assert(all(diff(t)>=0), 'Tiden går bakåt: %s', namn);
        context.signals.(namn) = struct('t_s',t, 'y',y, ...
            'unit',spec{j,3}, 'source_path',spec{j,2});
        fprintf('  %-30s %6d samples, %s\n', namn, numel(t), spec{j,3});
    end

    % Försök även bevara den ursprungliga hastighetsprofilen och dess enhet.
    % Denna fil laddas redan av init_nmpc. En tom enhet fylls inte med en gissning.
    hastnamn = 'ElectricVehicleThermalManagementWithHeatPumpVehicleSpeed.mat';
    kandidater = {fullfile(projektmapp,'Data',hastnamn), ...
                  fullfile(projektmapp,hastnamn), which(hastnamn)};
    context.drive_cycle_available = false;
    for j = 1:numel(kandidater)
        if ~isempty(kandidater{j}) && isfile(kandidater{j})
            V = load(kandidater{j}, 'DriveCycle');
            assert(isfield(V,'DriveCycle'), 'DriveCycle saknas i hastighetsfilen');
            ts = V.DriveCycle{1};
            context.drive_cycle = struct('t_s',double(ts.Time(:)), ...
                'values_native',double(ts.Data(:)), ...
                'unit',char(ts.DataInfo.Units), 'source_file',hastnamn);
            context.drive_cycle_available = true;
            break
        end
    end
    if ~context.drive_cycle_available
        warning('NN:MissingSpeedProfile', ...
            'Hastighetsfilen hittades inte. Kontexten sparas utan hastighetsprofil.');
    end

    % Spara bara vanlig numerik/text/strukturer, så att SciPy kan läsa resultatet.
    % Varje körning får en egen fil; inga rader från olika körningar blandas.
    utfil = fullfile(utdata, sprintf('context_rule_T%d.mat', abs(T_amb_C)));
    save(utfil, 'context', '-v7');
    fprintf('Saved: %s\n', utfil);
    clear inlast context
end
fprintf('\nThree context exports completed:\n%s\n', utdata);
end
