function outFile = run_scheduler_comparison(modes, temps, runRule)
%RUN_SCHEDULER_COMPARISON  Closed-loop test: NMPC with interpolated vs learned parameters.
%
%   run_scheduler_comparison                       % all nn_*.mat found + 'interp', at -5/-7/-10 degC
%   run_scheduler_comparison({'interp','george'}, -7)   % only the unseen -7 degC case (faster)
%   run_scheduler_comparison(modes, temps, false)  % skip the rule-based baseline runs
%
%   For every scheduler mode and ambient temperature it runs the full Simulink
%   simulation through RunThermalComparison, with a clean reset between runs
%   (persistent MPC state, warm-start cache, parameter interpolants), and stores
%   energy, temperatures, solver timing and the parameter trajectories (gamma_log).
%   Results are saved after every run, so an interrupted batch is not lost.
%   At the end compare_schedulers is called to make plots and the summary table.
%
%   Requires: init_nmpc run once (CasADi, Stuned, resampled in base workspace),
%   and the one-line change in MPCController_step_coreLQRNew.m (prepParaVec -> get_tuneS).

if nargin < 1 || isempty(modes)
    d = dir('nn_*.mat');
    modes = [{'interp'}, cellfun(@(s) s(4:end-4), {d.name}, 'UniformOutput', false)];
end
if ischar(modes) || isstring(modes), modes = cellstr(modes); end
if nargin < 2 || isempty(temps), temps = [-5, -7, -10]; end
if nargin < 3, runRule = true; end

if ~evalin('base', 'exist(''Stuned'',''var'')')
    fprintf('[runner] Stuned not in base workspace -> running init_nmpc\n');
    init_nmpc();
end
src = fileread(which('MPCController_step_coreLQRNew'));
if ~contains(src, 'get_tuneS(')
    error('runner:noSwitch', ['MPCController_step_coreLQRNew.m still calls prepParaVec in the per-step part.\n' ...
           'Replace line ~547 with:\n  tuneS = get_tuneS(x_meas_pk, Tamb_C, batImeas, speed, Stuned.tunedVariables);']);
end

outDir = fullfile(pwd, 'Results');
if ~exist(outDir, 'dir'), mkdir(outDir); end
outFile = fullfile(outDir, ['scheduler_comparison_' datestr(now, 'yyyy_mm_dd_HH_MM') '.mat']);
S = struct(); failed = {};
fprintf('[runner] Modes: %s | temperatures: %s degC | results -> %s\n', ...
        strjoin(modes, ', '), num2str(temps), outFile);

if runRule
    for T = temps
        reset_all();
        RunThermalComparison('rule', T);
        R = evalin('base', 'comparison_results');
        S.rule.(tag(T)) = R.rule.(tag(T));
        save(outFile, 'S', 'modes', 'temps', 'failed');
    end
end

for m = 1:numel(modes)
    mode = modes{m};
    fld = matlab.lang.makeValidName(['nmpc_' mode]);
    for T = temps
        reset_all();
        assignin('base', 'SCHED_MODE', mode);
        fprintf('\n[runner] ===== NMPC with scheduler ''%s'' at %d degC =====\n', mode, T);
        try
            RunThermalComparison('nmpc', T);
            R = evalin('base', 'comparison_results');
            st = R.nmpc.(tag(T));
            if evalin('base', 'exist(''gamma_log'',''var'')')
                st.gamma_log = evalin('base', 'gamma_log');
            end
            % RunThermalComparison overwrites its own timing field, so take it from the base workspace
            if evalin('base', 'exist(''mpc_timing'',''var'')')
                mt = evalin('base', 'mpc_timing');
                if isfield(mt, 'summary'), st.mpc_timing = mt.summary; end
            end
            st.sched_mode = mode;
            S.(fld).(tag(T)) = st;
        catch ME
            warning('runner:failed', 'Run %s at %d degC failed: %s', mode, T, ME.message);
            failed{end+1} = sprintf('%s @ %d degC: %s', mode, T, ME.message); %#ok<AGROW>
        end
        save(outFile, 'S', 'modes', 'temps', 'failed');
    end
end
evalin('base', 'SCHED_MODE = ''interp'';');
fprintf('\n[runner] Done. Saved %s\n', outFile);
if ~isempty(failed), fprintf('[runner] Failed runs:\n  %s\n', strjoin(failed, '\n  ')); end
compare_schedulers(outFile);
end

% -------------------------------------------------------------------------
function t = tag(T)
t = sprintf('T%d', abs(T));
end

function reset_all()
% Fresh controller for every run: persistent MPC/NLP state, supervisor filters,
% parameter interpolants and the scheduler log. The NMPC warm-start cache file is
% deleted so one scheduler's solution cannot initialise the next run.
clear get_tuneS prepParaVec thermalLogicAndEstimation
clear MPCController_step_coreLQRNew MPCController_step_coreLQRNew_v4
cache = fullfile(pwd, 'nmpc_warm_cache_v4.mat');
if isfile(cache), delete(cache); end
evalin('base', 'clear gamma_log mpc_timing comparison_results');
end
