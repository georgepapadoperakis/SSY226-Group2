function export_com_for_python()
% Placera i projektets huvudmapp. Exportera endast COM, inte NMPC-optimeraren.
% Ingen simulering eller CoolProp-beräkning behövs för att skapa symboliska F.
projektmapp = fileparts(mfilename('fullpath'));
addpath(fullfile(projektmapp, 'Source Files'));
addpath(fullfile(projektmapp, 'casadi-3.7.1-windows64-matlab2018b'));
assert(exist('casadi.SX', 'class') == 8, 'CasADi finns inte på MATLAB-sökvägen');
utdata = fullfile(projektmapp, 'Data', 'NN_training', 'COM');
if ~isfolder(utdata), mkdir(utdata); end

% Återanvänd exakt er befintliga modell. Vi kopierar inte dess ekvationer.
[F, ordning] = create_dynamics_nmpcNew();
F.save(fullfile(utdata, 'vehicle_com.casadi'));

metadata = struct();
metadata.sample_time_s = 1;
metadata.prediction_horizon = 30;
metadata.rk4_substeps = 1;
metadata.casadi_version = casadi.CasadiMeta.version();
metadata.state_names = {'motT_K','invT_K','dcdcT_K','batSoc','batT_K', ...
    'compPin_Pa','compPout_Pa','cabIntT_K','cabAirT_K'};
metadata.feature_names = [{'Tamb_C'}, metadata.state_names, {'Ib_A'}];
metadata.control_names = ordning.inputSignal;
metadata.fixed_parameter_names = [ordning.estimatedParams, ordning.controlOut];
metadata.gamma_names = ordning.tunePara;
assert(F.size1_in(2) == numel(metadata.fixed_parameter_names)+11, ...
    'Parameterordningen avviker från träningsbryggans antagande');
file = fullfile(utdata, 'com_metadata.json');
fid = fopen(file, 'w', 'n', 'UTF-8');
assert(fid ~= -1, 'Kunde inte öppna %s', file);
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(metadata, PrettyPrint=true));
fprintf('COM och metadata sparade i:\n%s\n', utdata);
end
