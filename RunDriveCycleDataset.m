function summary = RunDriveCycleDataset(varargin)
%RUNDRIVECYCLEDATASET  Build the drive-cycle dataset in one command (rule-based controller).
%
%   RunDriveCycleDataset                                   NEDC, UDDS, US06 at all 9 temperatures (27 runs)
%   RunDriveCycleDataset('DryRun', true)                   only list what would run
%   RunDriveCycleDataset('Cycles', 'all')                  also the original WLTCHW cycle (36 runs)
%   RunDriveCycleDataset('Cycles', {'UDDS'}, 'Temps', [-10 -7])
%   RunDriveCycleDataset('SkipExisting', false)            re-run even if an export already exists
%   RunDriveCycleDataset('SaveSimlog', true)               also keep the ~375 MB simlog per run
%
%   Every run calls RunThermalComparison('rule', T, cycle, saveSimlog), which writes
%       Results/run_<time>_<cycle>/training_data_rule_T<|T|>_<cycle>.mat
%   Runs whose export already exists are skipped, so an interrupted batch continues where it
%   stopped when started again. A failed run is reported and the next one still runs.
%   Each export is checked (length, missing signals, NaN, speed equals the cycle); the summary is
%   printed and saved as Results/drive_cycle_dataset_<time>.csv.
%
%   Cycles: see drive_cycle_list. The Simulink model file is never modified.

    p = inputParser;
    p.addParameter('Cycles', {'NEDC', 'UDDS', 'US06'});
    p.addParameter('Temps', 'all');
    p.addParameter('SaveSimlog', false, @islogical);
    p.addParameter('SkipExisting', true, @islogical);
    p.addParameter('DryRun', false, @islogical);
    p.parse(varargin{:});
    o = p.Results;

    repoRoot = fileparts(mfilename('fullpath'));
    addpath(repoRoot);
    addpath(fullfile(repoRoot, 'Source Files'));
    addpath(fullfile(repoRoot, 'Data'));

    all_temps = [0, -5, -7, -10, -12, -15, -20, -25, -30];
    if (ischar(o.Temps) || isstring(o.Temps)) && strcmpi(o.Temps, 'all'), temps = all_temps; else, temps = o.Temps(:)'; end
    if (ischar(o.Cycles) || isstring(o.Cycles)) && isscalar(string(o.Cycles)) && strcmpi(o.Cycles, 'all')
        cycles = {drive_cycle_list().handle};
    else
        cycles = cellfun(@(c) drive_cycle_list(c).handle, cellstr(string(o.Cycles)), 'UniformOutput', false);
    end

    % ---- plan
    plan = {};
    for i = 1:numel(cycles)
        for T = temps
            existing = find_exports(repoRoot, cycles{i}, T, 0);
            todo = ~(o.SkipExisting && ~isempty(existing));
            plan(end+1, :) = {cycles{i}, T, todo}; %#ok<AGROW>
        end
    end
    n_todo = sum([plan{:, 3}]);
    fprintf('\n[Dataset] %d cycle(s) x %d temperature(s): %d to run, %d already exported (skipped)\n', ...
            numel(cycles), numel(temps), n_todo, size(plan, 1) - n_todo);
    for i = 1:numel(cycles)
        rows = plan(strcmp(plan(:, 1), cycles{i}), :);
        run_T = [rows{[rows{:, 3}], 2}];
        fprintf('   %-6s run: %-40s skip: %s\n', cycles{i}, mat2str(run_T), mat2str([rows{~[rows{:, 3}], 2}]));
    end
    if o.DryRun || n_todo == 0
        if o.DryRun, fprintf('[Dataset] DryRun: nothing simulated.\n'); else, fprintf('[Dataset] Nothing to do.\n'); end
        summary = table();
        return
    end

    % ---- simulations
    out = {};
    t_all = tic; k = 0;
    for r = 1:size(plan, 1)
        [cyc, T, todo] = plan{r, :};
        if ~todo, continue; end
        k = k + 1;
        fprintf('\n[Dataset] ===== run %d/%d: %s at %d degC  (%.0f min elapsed) =====\n', k, n_todo, cyc, T, toc(t_all) / 60);
        t0 = tic; started = now; status = "ok"; note = "";
        try
            RunThermalComparison('rule', T, cyc, o.SaveSimlog);
        catch ME
            status = "FAILED"; note = string(ME.message);
            fprintf(2, '[Dataset] %s at %d degC failed: %s\n', cyc, T, ME.message);
        end
        [file, chk] = check_export(repoRoot, cyc, T, started);
        if status == "ok" && chk ~= "ok", status = "CHECK"; end
        out(end+1, :) = {string(cyc), T, status, toc(t0) / 60, string(file), strtrim(chk + " " + note)}; %#ok<AGROW>
        close all force                                    % plots are saved to disk; keep memory flat
    end

    % ---- summary
    summary = cell2table(out, 'VariableNames', {'cycle', 'T_amb_C', 'status', 'minutes', 'file', 'check'});
    fprintf('\n[Dataset] Done: %d run(s) in %.0f min\n', height(summary), toc(t_all) / 60);
    disp(summary(:, {'cycle', 'T_amb_C', 'status', 'minutes', 'check'}));
    csv = fullfile(repoRoot, 'Results', ['drive_cycle_dataset_' datestr(now, 'yyyy_mm_dd_HH_MM') '.csv']);
    writetable(summary, csv);
    fprintf('[Dataset] Summary saved: %s\n', csv);
end


function f = find_exports(repoRoot, cyc, T, since)
% Rule-based exports for this cycle and temperature (newer than 'since', a datenum).
% Exports from before 18 Nov have no cycle in the name; they are WLTCHW.
    f = dir(fullfile(repoRoot, 'Results', '**', sprintf('training_data_rule_T%d_%s.mat', abs(T), cyc)));
    if strcmp(cyc, 'WLTCHW')
        f = [f; dir(fullfile(repoRoot, 'Results', '**', sprintf('training_data_rule_T%d.mat', abs(T))))];
    end
    f = f([f.datenum] >= since);
end


function [file, chk] = check_export(repoRoot, cyc, T, started)
% The export written by this run, and a few sanity checks on it.
    f = find_exports(repoRoot, cyc, T, started - 1/86400);
    if isempty(f)
        file = ""; chk = "no export file from this run";
        return
    end
    [~, k] = max([f.datenum]);
    file = fullfile(f(k).folder, f(k).name);
    S = load(file);
    d = S.trainingData;
    V = load(fullfile(repoRoot, 'DriveCycles_SSY226.mat'), cyc);
    ref = V.(cyc){1};
    v_ref = interp1(ref.Time(:), double(ref.Data(:)), d.t_s, 'linear', 0);
    problems = strings(0);
    if abs(d.t_s(end) - ref.Time(end)) > 1
        problems(end+1) = sprintf("length %.0f s (cycle %.0f s)", d.t_s(end), ref.Time(end));
    end
    if max(abs(d.vehSpeed_kmh - v_ref)) > 0.5
        problems(end+1) = "speed differs from the cycle";
    end
    if ~isempty(S.missing)
        problems(end+1) = "missing: " + strjoin(string(S.missing), ", ");
    end
    if any(~isfinite(d.compPout_MPa)) || any(~isfinite(d.cabAirT_C))
        problems(end+1) = "NaN in states";
    end
    if isempty(problems), chk = "ok"; else, chk = strjoin(problems, "; "); end
end
