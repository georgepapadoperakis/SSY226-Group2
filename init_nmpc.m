function init_nmpc()

repoRoot = fileparts(mfilename('fullpath'));
addpath(repoRoot);
addpath(fullfile(repoRoot, 'Data'));
% ── CasADi + IPOPT linear solver ──────────────────────────────────────────
% USE_HSL = true: MA97 (as in the paper). CoinHSL is a personal academic licence, so it is kept OUTSIDE the repo,
% next to it in Project_LP2\:
%   CoinHSL\CoinHSL.v*\bin\libhsl.dll   - the unpacked CoinHSL Windows binaries
%   casadi-3.7.1-hsl\                     - private CasADi copy whose GCC 11.2 runtime DLLs are replaced by
%                                           CoinHSL's GCC 13.2 ones (with the repo's CasADi, loading MA97
%                                           crashes MATLAB: both runtimes cannot be loaded at once)
% If either is missing (e.g. on a group member's computer), the repo's CasADi and MUMPS are used.
% Switching between the two CasADi folders needs a MATLAB restart (the CasADi MEX file stays loaded).
USE_HSL   = true;
casadiRepo = fullfile(repoRoot, 'casadi-3.7.1-windows64-matlab2018b');
casadiHsl  = fullfile(fileparts(repoRoot), 'casadi-3.7.1-hsl');
hslDirs    = dir(fullfile(fileparts(repoRoot), 'CoinHSL', 'CoinHSL.v*', 'bin', 'libhsl.dll'));
if USE_HSL && isfolder(casadiHsl) && ~isempty(hslDirs)
    loaded = inmem('-completenames');
    if any(contains(loaded, casadiRepo))
        warning('init_nmpc:restart', ['The repo''s CasADi is already loaded in this MATLAB session. ', ...
                'Restart MATLAB to use MA97; using MUMPS for now.']);
        addpath(casadiRepo);  IPOPT_LINEAR_SOLVER = 'mumps';
    else
        if contains(path, casadiRepo), rmpath(casadiRepo); end
        addpath(casadiHsl);
        hslBin = hslDirs(end).folder;
        if ~contains(getenv('PATH'), hslBin), setenv('PATH', [hslBin ';' getenv('PATH')]); end
        IPOPT_LINEAR_SOLVER = 'ma97';
        fprintf('[init_nmpc] CasADi from %s, IPOPT linear solver ma97\n', casadiHsl);
    end
else
    addpath(casadiRepo);  IPOPT_LINEAR_SOLVER = 'mumps';
    fprintf('[init_nmpc] IPOPT linear solver mumps (USE_HSL = %d)\n', USE_HSL);
end
assignin('base', 'IPOPT_LINEAR_SOLVER', IPOPT_LINEAR_SOLVER);

if ~exist('casadi.SX', 'class')
    error('NMPC:missingDependency', ...
          ['CasADi is required but not found on the MATLAB path.\n', ...
           'Download it from: https://web.casadi.org/get/\n', ...
           'Then add it using: addpath(''<your_casadi_folder> in the init_nmpc file'')']);
end

try
    val = py.CoolProp.CoolProp.PropsSI('C', 'T', 273.15, 'P', 101325, 'Air');
    if ~isfinite(val) || val <= 0
        error('CoolProp returned an invalid value.');
    end
    fprintf('CoolProp is installed and working (cp_air = %.2f J/kg/K)\n', val);
catch ME
    error('NMPC:missingDependency', ...
          ['CoolProp Python wrapper not found or not working.\n', ...
           'Ensure CoolProp is installed in your Python environment:\n', ...
           '    pip install CoolProp\n', ...
           'Also verify MATLAB is linked to the correct Python:\n', ...
           '    pyenv\n\n', ...
           'Original error: %s'], ME.message);
end
% 2) Load tuning/resampled data used by MPC (if you still rely on it)
S = load('tuning_data_10.mat','resampled');
assignin('base','resampled', S.resampled);

Stuned = load('tunedVariables_minus5_minus10.mat');
assignin('base','Stuned', Stuned);


% ── Prediction model of the NMPC (COM + where its tuning parameters come from) ──────────────────────────
%   'table'  original COM, 11 tuning parameters from the interpolation table prepParaVec (paper setup)
%   'nn11'   original COM, 11 tuning parameters from the gamma network  (python/models/gamma11_original_com.mat)
%   'nn15'   fixed COM,    15 tuning parameters from the gamma network  (python/models/gamma15_fixed_com.mat)
% RunThermalComparison sets COM_MODEL in the base workspace (from its comModel argument or this line) and then runs
% init_nmpc, so that value wins. When you run init_nmpc by hand after editing this line, first: clear COM_MODEL
COM_MODEL = 'nn15';
if evalin('base', 'exist(''COM_MODEL'', ''var'')'), COM_MODEL = evalin('base', 'COM_MODEL'); end
switch lower(COM_MODEL)
    case 'table', COM_FIXES = false; USE_GAMMANET = false;
    case 'nn11',  COM_FIXES = false; USE_GAMMANET = true;
    case 'nn15',  COM_FIXES = true;  USE_GAMMANET = true;
    otherwise, error('init_nmpc:badModel', 'COM_MODEL must be ''table'', ''nn11'' or ''nn15'' (got ''%s'').', COM_MODEL);
end
assignin('base', 'COM_MODEL', lower(COM_MODEL));
assignin('base', 'COM_FIXES', COM_FIXES);          % used by create_dynamics_nmpcNew.m and gamma_network.m
assignin('base', 'USE_GAMMANET', USE_GAMMANET);    % used by MPCController_step_coreLQRNew.m
fprintf('[init_nmpc] NMPC prediction model: %s\n', lower(COM_MODEL));

N  = 30;     % prediction horizon [steps]
Ts = 1;      % MPC sample time [s]
nx = 9;      % number of states
nu = 6;      % number of control inputs

MPC_INPUT_LEN = nx + 6*N + 3;

N_MAX = N;   % 
assignin('base', 'N_MAX', N_MAX);

ControlSampleTime = Ts; 


if N > N_MAX
    error('init_nmpc:HorizonTooLarge', ...
          'N=%d exceeds N_MAX=%d. Increase N_MAX in init_nmpc.m', N, N_MAX);
end


assignin('base', 'N',             N);
assignin('base', 'Ts',            Ts);
assignin('base', 'nx',            nx);
assignin('base', 'nu',            nu);
assignin('base', 'MPC_INPUT_LEN', MPC_INPUT_LEN);


fprintf('[init_nmpc] ✓ MPC parameters set: N=%d, Ts=%.1fs, nx=%d, nu=%d\n', ...
        N, Ts, nx, nu);
fprintf('[init_nmpc]   Input vector length: MPC_INPUT_LEN = %d\n', MPC_INPUT_LEN);


V = load('ElectricVehicleThermalManagementWithHeatPumpVehicleSpeed.mat');
DC = V.DriveCycle;
speed_ts = DC{1};          
t_grid = speed_ts.Time(:);
v_grid = speed_ts.Data(:);
a_grid = gradient(v_grid, t_grid);

assignin('base','t_grid', t_grid);
assignin('base','v_grid', v_grid);
assignin('base','a_grid', a_grid);

disp('[init_nmpc] Loaded CasADi + data + speed profile.');
disp('*** init_nmpc ran ***');

%% ── 5. Summary ───────────────────────────────────────────────────────────
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║           init_nmpc — Initialization Summary    ║\n');
fprintf('╠══════════════════════════════════════════════════╣\n');
fprintf('║  Horizon  N  : %d steps                          \n', N);
fprintf('║  Sample time : %.1f s                            \n', Ts);
fprintf('║  States   nx : %d                                \n', nx);
fprintf('║  Inputs   nu : %d                                \n', nu);
fprintf('║  Input vec   : %d elements                       \n', MPC_INPUT_LEN);
fprintf('╚══════════════════════════════════════════════════╝\n');

% Confirm key workspace variables are present
whos('N', 'Ts', 'nx', 'nu', 'MPC_INPUT_LEN', 'Stuned', 'resampled');
end