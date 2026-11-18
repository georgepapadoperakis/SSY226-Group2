function export_logs_to_python()
% EXPORT_LOGS_TO_PYTHON: steg 1, fran sparad Simscape-logg till numerisk data.
%
% Placera denna fil i SSY226-Group2, bredvid RunThermalComparison.m.
% Kor funktionen i MATLAB genom att skriva: export_logs_to_python
%
% Koden anvander redan sparade korningar; den startar ingen simulering.
% Sokkedjorna ar hamtade fran compute_energy_results i er uppladdade fil.
% Vi behaller varje signals egen tidsvektor och gor ingen omsampling.

%% 1. Hitta projektet och valj den befintliga korningen
% mfilename('fullpath') ger denna fils plats, oberoende av Current Folder.
projektmapp = fileparts(mfilename('fullpath'));
korning = 'run_2026_11_11_10_22_21';
indata = fullfile(projektmapp, 'Results', korning);
utdata = fullfile(projektmapp, 'Data', 'NN_exports', korning);

assert(isfolder(indata), 'Hittar inte resultatmappen: %s', indata);
if ~isfolder(utdata)
    mkdir(utdata);
end

%% 2. Beskriv vilka loggade signaler som ska exporteras
% Varje rad innehaller: nytt namn, exakt nodsokvag, onskad enhet.
% Batteripacken sparas separat. Ett medelvarde kraver forst att deras
% tidsstamplar ar jamforbara; vi antar inte att lika langd betyder samma tid.
spec = {
    'T_bat_pack1_C', 'Battery.Pack_1.Thermal_Model.T', 'degC';
    'T_bat_pack2_C', 'Battery.Pack_2.Thermal_Model.T', 'degC';
    'T_bat_pack3_C', 'Battery.Pack_3.Thermal_Model.T', 'degC';
    'T_bat_pack4_C', 'Battery.Pack_4.Thermal_Model.T', 'degC';
    'T_cabin_C', 'Cabin.Cabin_Air_Volume.T_I', 'degC';
    'T_motor_C', 'Motor.Motor.T', 'degC';
    'T_inverter_C', 'Inverter.Inverter.T', 'degC';
    'T_dcdc_C', 'DCDC.DC_DC_Converter.T', 'degC';
    'P_comp_mech_W', 'Compressor.Compressor.mechanical_power', 'W';
    'P_motor_pump_mech_W', 'Motor_Pump.Motor_Pump.mechanical_power', 'W';
    'P_battery_pump_mech_W', 'Battery_Pump.Battery_Pump.mechanical_power', 'W';
    'P_blower_logged_W', 'Blower.Blower.power', 'W';
    'P_fan_mech_W', 'Radiator.Fan.mechanical_power', 'W';
    'Q_heater_W', 'Heater.Controlled_Heat_Flow_Rate_Source.Q', 'W';
};

%% 3. Oppna varje fil och skapa en vanlig MATLAB-struktur
temperaturer = [-5, -7, -10];
for T_amb_C = temperaturer
    kallfil = fullfile(indata, sprintf('simlog_rule_T%d.mat', abs(T_amb_C)));
    assert(isfile(kallfil), 'Hittar inte filen: %s', kallfil);
    fprintf('\nLaser: %s\n', kallfil);

    % load med ett utargument samlar variablerna i en struktur.
    inlast = load(kallfil, 'simlog');
    assert(isfield(inlast, 'simlog'), 'Variabeln simlog saknas i %s', kallfil);
    logg = inlast.simlog;

    % Metadata beskriver data; det ar inte extra traningssignaler.
    run_data = struct();
    run_data.schema_version = 1;
    run_data.T_amb_C = T_amb_C;
    run_data.controller = 'rule';
    run_data.source_file = kallfil;
    run_data.source_run = korning;
    run_data.sampling = 'native; no resampling';
    run_data.signals = struct();

    for j = 1:size(spec, 1)
        namn = spec{j, 1};
        sokvag = spec{j, 2};
        enhet = spec{j, 3};
        signal = las_skalarsignal(logg, sokvag, enhet);
        % Dynamiskt faltnamn: variabeln namn bestammer faltets namn.
        run_data.signals.(namn) = signal;
        fprintf('  %-27s %7d prov, %.3f till %.3f s\n', ...
            namn, numel(signal.t_s), signal.t_s(1), signal.t_s(end));
    end

    %% 4. Bevara aven originalkodens elektriska effektuppskattningar
    % Dessa ar BERAKNADE fran loggade effekter och antagna verkningsgrader.
    % De ar inte separat loggade elektriska matningar eller styrkommandon.
    antaganden = {
        'P_comp_elec_est_W', 'P_comp_mech_W', 0.90*0.90;
        'P_motor_pump_elec_est_W', 'P_motor_pump_mech_W', 0.88;
        'P_battery_pump_elec_est_W', 'P_battery_pump_mech_W', 0.88;
        'P_blower_elec_est_W', 'P_blower_logged_W', 0.80;
        'P_fan_elec_est_W', 'P_fan_mech_W', 0.85*0.95;
    };
    for j = 1:size(antaganden, 1)
        nytt_namn = antaganden{j, 1};
        kallnamn = antaganden{j, 2};
        eta = antaganden{j, 3};
        signal = run_data.signals.(kallnamn);
        signal.y = signal.y / eta;
        signal.kind = 'derived estimate';
        signal.derivation = sprintf('%s / %.12g', kallnamn, eta);
        signal.efficiency_assumed = eta;
        run_data.signals.(nytt_namn) = signal;
    end

    %% 5. Spara en ny fil som SciPy kan lasa direkt
    % -v7 valjs uttryckligen. Filen innehaller bara numerik, text och strukturer.
    % Samma exportfil ersatts om funktionen kors igen; kalloggen andras inte.
    utfil = fullfile(utdata, sprintf('rule_T%d_numeric.mat', abs(T_amb_C)));
    save(utfil, 'run_data', '-v7');
    fprintf('Sparad: %s\n', utfil);
    clear inlast logg run_data signal
end
fprintf('\nKlart. Tre numeriska filer finns i:\n%s\n', utdata);
end


function signal = las_skalarsignal(logg, sokvag, enhet)
% En lokal hjalpfunktion samlar aterkommande arbete pa ett stalle.
% Vi anvander samma nod-ID:n som originalkodens punktnotation.
delar = strsplit(sokvag, '.');
nod = logg;
for k = 1:numel(delar)
    nod = nod.(delar{k});
end
serie = nod.series;

% (:), efter time/values, gor varje vektor till en kolumn.
% values omvandlar till den enhet som angavs i specifikationen.
t = time(serie);
y = values(serie, 'Units', enhet);
assert(isvector(t) && isvector(y), ...
    'Signal %s ar inte en skalar tidsserie.', sokvag);
t = double(t(:));
y = double(y(:));

% Stoppa vid saknad/ogiltig data; skapa inte tysta NaN-ersattningar.
assert(numel(t) >= 2 && numel(t) == numel(y), ...
    'Fel antal tidsstamplar eller varden: %s', sokvag);
assert(all(isfinite(t)) && all(isfinite(y)), ...
    'NaN eller Inf finns i signalen: %s', sokvag);
assert(all(diff(t) >= 0), 'Tiden gar bakat: %s', sokvag);
assert(t(end) > t(1), 'Signalens tidsintervall ar tomt: %s', sokvag);

% Lika tidsstamplar kan forekomma vid handelser och bevaras i denna export.
signal = struct('t_s', t, 'y', y, 'unit', enhet, ...
    'source_path', sokvag, 'kind', 'logged; converted to stated unit');
end
