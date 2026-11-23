function RunAllComModels(temps, cycle, models)
%RUNALLCOMMODELS  Closed-loop comparison: rule-based controller + NMPC with each prediction model.
%
%   RunAllComModels                         WLTCHW at -10 °C: rule, NMPC-table, NMPC-nn11, NMPC-nn15 (~1 h 45 min)
%   RunAllComModels(temps)                  e.g. -7 or [-10 -7]
%   RunAllComModels(temps, cycle)           cycle: 'WLTCHW' (paper), 'NEDC', 'UDDS' or 'US06'
%   RunAllComModels(temps, cycle, models)   models: subset of {'table', 'nn11', 'nn15'}
%
%   Prediction models of the NMPC:
%     table  original COM, 11 tuning parameters from the interpolation table (the paper's setup)
%     nn11   original COM, 11 tuning parameters from the gamma network (python/models/gamma11_original_com.mat)
%     nn15   fixed COM,    15 tuning parameters from the gamma network (python/models/gamma15_fixed_com.mat)
%
%   Each run gets its own folder, e.g. Results/run_<time>_WLTCHW_rule, Results/run_<time>_WLTCHW_nmpc-nn15.
%   Plots of all runs together (energy, cabin, battery, inputs), from the python folder:
%       python make_figures_nmpc.py
%   Run time: rule ~4 min, each NMPC ~30 min (one IPOPT solve per second over the 1-h drive).

    if nargin < 1 || isempty(temps),  temps  = -10;      end
    if nargin < 2 || isempty(cycle),  cycle  = 'WLTCHW'; end
    if nargin < 3 || isempty(models), models = {'table', 'nn11', 'nn15'}; end
    if ischar(models), models = {models}; end

    repoRoot = fileparts(mfilename('fullpath'));
    cd(repoRoot);                                  % warm-start caches and Results/ are relative to the repo
    t0 = tic;
    fprintf('\n[RunAllComModels] %s at %s °C: rule + NMPC with %s\n', cycle, mat2str(temps), strjoin(models, ', '));

    RunThermalComparison('rule', temps, cycle, false);
    for i = 1:numel(models)
        fprintf('\n[RunAllComModels] NMPC %d/%d: %s (%.0f min so far)\n', i, numel(models), models{i}, toc(t0) / 60);
        RunThermalComparison('nmpc', temps, cycle, false, models{i});
    end
    fprintf('\n[RunAllComModels] all runs done in %.0f min. Plots: python make_figures_nmpc.py (in the python folder)\n', ...
            toc(t0) / 60);
end
