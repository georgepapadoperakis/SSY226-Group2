function T_summary = compare_schedulers(file)
%COMPARE_SCHEDULERS  Plots and summary table for run_scheduler_comparison results.
%
%   compare_schedulers('Results/scheduler_comparison_<date>.mat')
%   compare_schedulers            % newest scheduler_comparison_*.mat in Results/
%
%   Per ambient temperature it makes:
%     - cabin air, battery and motor temperature for every method
%     - total thermal power (60 s moving average)
%     - energy bar chart
%     - parameter trajectories p1..p11: each network vs the interpolated values
%   and a summary table (printed, and saved as .csv and .tex):
%     total energy, saving vs rule-based and vs NMPC-interpolation, time until
%     the cabin reaches 20 degC, mean |cabin - 21 degC| after 10 min, minimum
%     battery temperature, mean solver time, share of OOD fallback steps.

if nargin < 1 || isempty(file)
    d = dir(fullfile(pwd, 'Results', 'scheduler_comparison_*.mat'));
    if isempty(d), error('No scheduler_comparison_*.mat found in Results/'); end
    [~, i] = max([d.datenum]); file = fullfile(d(i).folder, d(i).name);
end
L = load(file); S = L.S; temps = L.temps;
[outDir, base] = fileparts(file);
figDir = fullfile(outDir, base); if ~exist(figDir, 'dir'), mkdir(figDir); end

methods = fieldnames(S)';
order = [intersect({'rule'}, methods), intersect({'nmpc_interp'}, methods), ...
         setdiff(methods, {'rule', 'nmpc_interp'}, 'stable')];
methods = order;
label = @(m) strrep(strrep(strrep(m, 'nmpc_', 'NMPC + '), 'rule', 'Rule-based'), 'interp', 'interpolation');
rows = {};

for T = temps
    tg = sprintf('T%d', abs(T));
    have = methods(cellfun(@(m) isfield(S.(m), tg), methods));
    if isempty(have), continue; end
    C = lines(numel(have));

    % ---- temperatures
    fig = figure('Visible', 'off', 'Position', [100 100 1300 380]);
    sig = {'t_cabAir_C', 'T_cabAir_C', 'Cabin air'; 't_bat_C', 'T_bat_C', 'Battery'; 't_mot_C', 'T_mot_C', 'Motor'};
    for s = 1:3
        subplot(1, 3, s); hold on; grid on;
        for j = 1:numel(have)
            st = S.(have{j}).(tg);
            if isfield(st, sig{s, 1})
                plot(st.(sig{s, 1}), st.(sig{s, 2}), 'Color', C(j, :), 'LineWidth', 1.3, 'DisplayName', label(have{j}));
            end
        end
        if s == 1, yline(21, 'k:', 'HandleVisibility', 'off'); end
        xlabel('Time [s]'); ylabel('Temperature [\circC]'); title(sprintf('%s at %d\\circC', sig{s, 3}, T));
        if s == 1, legend('Location', 'southeast'); end
    end
    save_fig(fig, figDir, sprintf('temperatures_%s', tg));

    % ---- power and energy
    fig = figure('Visible', 'off', 'Position', [100 100 1200 400]);
    subplot(1, 2, 1); hold on; grid on;
    for j = 1:numel(have)
        st = S.(have{j}).(tg);
        [t1, w1] = to1hz(st.t, st.W_total);
        plot(t1, movmean(w1, 60), 'Color', C(j, :), 'LineWidth', 1.2, 'DisplayName', label(have{j}));
    end
    xlabel('Time [s]'); ylabel('Total TEM power [W] (60 s mean)'); title(sprintf('Power at %d\\circC', T)); legend;
    subplot(1, 2, 2);
    E = cellfun(@(m) S.(m).(tg).E_total_Wh, have);
    b = bar(categorical(cellfun(label, have, 'UniformOutput', false), cellfun(label, have, 'UniformOutput', false)), E);
    b.FaceColor = 'flat'; b.CData = C; grid on; ylabel('Energy [Wh]'); title(sprintf('Total energy at %d\\circC', T));
    save_fig(fig, figDir, sprintf('energy_%s', tg));

    % ---- parameter trajectories
    nnm = have(cellfun(@(m) isfield(S.(m).(tg), 'gamma_log') && ~strcmp(m, 'nmpc_interp') ...
                                   && ~isempty(S.(m).(tg).gamma_log), have));
    if ~isempty(nnm)
        fig = figure('Visible', 'off', 'Position', [50 50 1500 850]);
        for i = 1:11
            subplot(3, 4, i); hold on; grid on;
            g0 = S.(nnm{1}).(tg).gamma_log;
            plot(g0(:, 1), g0(:, 14 + i), 'k--', 'LineWidth', 1.1, 'DisplayName', 'interpolation');
            for j = 1:numel(nnm)
                g = S.(nnm{j}).(tg).gamma_log;
                plot(g(:, 1), g(:, 3 + i), 'LineWidth', 1.0, 'DisplayName', label(nnm{j}));
            end
            title(sprintf('p%d', i)); xlabel('MPC step [s]');
            if i == 1, legend('Location', 'best', 'FontSize', 7); end
        end
        sgtitle(sprintf('Scheduled COM parameters at %d\\circC (dashed: interpolation along the same run)', T));
        save_fig(fig, figDir, sprintf('parameters_%s', tg));
    end

    % ---- table rows
    E_rule = NaN; E_int = NaN;
    if isfield(S, 'rule') && isfield(S.rule, tg), E_rule = S.rule.(tg).E_total_Wh; end
    if isfield(S, 'nmpc_interp') && isfield(S.nmpc_interp, tg), E_int = S.nmpc_interp.(tg).E_total_Wh; end
    for j = 1:numel(have)
        st = S.(have{j}).(tg);
        [tc, Tc] = to1hz(st.t_cabAir_C, st.T_cabAir_C);
        i20 = find(Tc >= 20, 1);  t20 = NaN; if ~isempty(i20), t20 = tc(i20); end
        comfort = mean(abs(Tc(tc >= 600) - 21));
        solver = NaN; if isfield(st, 'mpc_timing') && isfield(st.mpc_timing, 'mean_solver_ms'), solver = st.mpc_timing.mean_solver_ms; end
        oodp = NaN; if isfield(st, 'gamma_log') && ~isempty(st.gamma_log), oodp = 100 * mean(st.gamma_log(:, 3)); end
        rows(end+1, :) = {T, label(have{j}), st.E_total_Wh, 100 * (1 - st.E_total_Wh / E_rule), ...
                          100 * (1 - st.E_total_Wh / E_int), t20, comfort, min(st.T_bat_C), solver, oodp}; %#ok<AGROW>
    end
end

T_summary = cell2table(rows, 'VariableNames', {'Tamb_C', 'Method', 'Energy_Wh', 'Saving_vs_rule_pct', ...
    'Saving_vs_NMPCinterp_pct', 'Time_to_20C_s', 'MeanAbsCabinErr_after10min_K', 'MinBatT_C', ...
    'MeanSolver_ms', 'OOD_fallback_pct'});
disp(T_summary);
writetable(T_summary, fullfile(figDir, 'summary.csv'));

fid = fopen(fullfile(figDir, 'summary.tex'), 'w');
fprintf(fid, '%% Auto-generated by compare_schedulers\n\\begin{tabular}{rlrrrrr}\n\\hline\n');
fprintf(fid, '$T_{amb}$ [$^\\circ$C] & Method & Energy [Wh] & vs rule [\\%%] & vs interp. [\\%%] & $t_{20^\\circ C}$ [s] & $|T_{cab}-21|$ [K]\\\\\n\\hline\n');
for r = 1:size(rows, 1)
    fprintf(fid, '%d & %s & %.0f & %.1f & %.1f & %.0f & %.2f\\\\\n', rows{r, 1}, rows{r, 2}, rows{r, 3}, ...
            rows{r, 4}, rows{r, 5}, rows{r, 6}, rows{r, 7});
end
fprintf(fid, '\\hline\n\\end{tabular}\n'); fclose(fid);
fprintf('[compare] Figures and tables saved in %s\n', figDir);
end

% -------------------------------------------------------------------------
function [t1, y1] = to1hz(t, y)
[t, iu] = unique(t(:)); y = y(:); y = y(iu);
t1 = (ceil(t(1)):floor(t(end)))';
y1 = interp1(t, y, t1, 'linear');
end

function save_fig(fig, d, name)
savefig(fig, fullfile(d, [name '.fig']));
exportgraphics(fig, fullfile(d, [name '.png']), 'Resolution', 200);
close(fig);
end
