function R = validate_com_fig2(modes, temps)
%VALIDATE_COM_FIG2  Replicates Fig. 2 / Table 1 of Lokur & Murgovski with any parameter scheduler.
%
%   validate_com_fig2                              % 'interp' + every nn_*.mat, at -10/-7/-5 degC
%   validate_com_fig2({'interp','george'}, -7)     % only the unseen -7 degC case
%
%   The group's control-oriented model (COM, create_dynamics_nmpcNew) is simulated
%   OPEN LOOP over the full drive cycle, driven by the same inputs the rule-based
%   controller applied to the high-fidelity Simscape model (compressor speed,
%   blower flow, pump speeds, heater power, fan speed) and the same disturbances
%   (heat loads, battery current, ambient). Its states are compared with the
%   Simscape reference, exactly as in the paper's model validation.
%   The only difference between the curves is where p1..p11 come from:
%     'interp'  -> prepParaVec (the paper's interpolation)
%     <name>    -> network nn_<name>.mat via get_tuneS
%   Thermophysical properties are re-evaluated each second from the COM's own
%   states (CoolProp), as in the MPC. No MPC / IPOPT is needed.
%
%   Needs: simlog_rule_T5/T7/T10_export.mat (from export_simlog_runs), init_nmpc,
%   CasADi, CoolProp. Output: Results/com_validation_<date>/ (figures, table, .mat)

root = pwd;
addpath(root, fullfile(root, 'Data'), fullfile(root, 'Source Files'));
if nargin < 1 || isempty(modes)
    d = [dir(fullfile(root, 'nn_*.mat')); dir(fullfile(root, 'Data', 'nn_*.mat'))];
    modes = [{'interp'}, unique(cellfun(@(s) s(4:end-4), {d.name}, 'UniformOutput', false))];
end
if ischar(modes) || isstring(modes), modes = cellstr(modes); end
if nargin < 2 || isempty(temps), temps = [-10, -7, -5]; end

if ~evalin('base', 'exist(''Stuned'',''var'')'), init_nmpc(); end
Stuned = evalin('base', 'Stuned');
TV = Stuned.tunedVariables;
[F, ~, ~, packParams] = create_dynamics_nmpcNew();
V = load('ElectricVehicleThermalManagementWithHeatPumpVehicleSpeed.mat');
speedTs = V.DriveCycle{1};

outDir = fullfile(root, 'Results', ['com_validation_' datestr(now, 'yyyy_mm_dd_HH_MM')]);
if ~exist(outDir, 'dir'), mkdir(outDir); end
R = struct(); rows = {};
paper = struct('T10', [1.60 1.35; 1.50 1.26; 2.12 1.61; 1.61 1.36; 1.04 0.58], ...
               'T7',  [1.70 1.46; 1.53 1.33; 2.00 1.62; 1.93 1.77; 1.45 1.14], ...
               'T5',  [1.43 1.18; 1.29 1.05; 1.83 1.30; 1.53 1.33; 1.01 0.81]);
stateNames = {'T_mot', 'T_inv', 'T_dcdc', 'T_b', 'T_cair', 'p_in', 'p_out'};
stateIdx   = [1, 2, 3, 5, 9, 6, 7];

for T = temps
    tg = sprintf('T%d', abs(T));
    D = load_run(sprintf('simlog_rule_%s_export.mat', tg), speedTs);
    fprintf('\n=== %d degC: %d s of reference data ===\n', T, numel(D.t));
    for m = 1:numel(modes)
        mode = modes{m};
        fld = matlab.lang.makeValidName(mode);
        fprintf('[%s] simulating COM open loop ...\n', mode);
        tic; [X, ood, nfail] = simulate_com(F, packParams, D, mode, TV); el = toc;
        R.(tg).(fld).X = X; R.(tg).(fld).ood = ood;
        err = X - D.xref;
        met = zeros(numel(stateIdx), 2);
        for s = 1:numel(stateIdx)
            e = err(stateIdx(s), :);
            if s >= 6, e = e / 1e5; end                     % pressures in bar
            ok = isfinite(e);
            met(s, :) = [sqrt(mean(e(ok).^2)), mean(abs(e(ok)))];
        end
        R.(tg).(fld).rmse_mae = met;
        fprintf('[%s] done in %.0f s (%d CoolProp fallbacks, OOD fallback on %.1f%% of steps)\n', ...
                mode, el, nfail, 100 * mean(ood));
        pr = NaN(5, 2); if isfield(paper, tg), pr = paper.(tg); end
        for s = 1:numel(stateIdx)
            ppr = [NaN NaN]; if s <= 5, ppr = pr(s, :); end
            rows(end+1, :) = {T, mode, stateNames{s}, met(s, 1), met(s, 2), ppr(1), ppr(2)}; %#ok<AGROW>
        end
    end
    R.(tg).ref = D;
    make_figures(R.(tg), D, modes, T, outDir);
end

Tab = cell2table(rows, 'VariableNames', {'Tamb_C', 'Scheduler', 'State', 'RMSE', 'MAE', ...
                                         'Paper_RMSE_interp', 'Paper_MAE_interp'});
disp('RMSE / MAE of the COM vs. the high-fidelity reference (temperatures degC, pressures bar):');
disp(Tab);
writetable(Tab, fullfile(outDir, 'table1_comparison.csv'));
save(fullfile(outDir, 'com_validation.mat'), 'R', 'Tab', 'modes', 'temps');
evalin('base', 'SCHED_MODE = ''interp'';');
fprintf('\nSaved figures, table and data in %s\n', outDir);
end

% =========================================================================
function D = load_run(file, speedTs)
if isempty(which(file)) && ~isfile(file)
    error('validate:noData', '%s not found (run export_simlog_runs first).', file);
end
M = load(file);
names = cellstr(M.names);
g = @(n) M.data(:, find(strcmp(names, ['simlog.' n]), 1));
need = @(n) assert(any(strcmp(names, ['simlog.' n])), 'Signal simlog.%s missing in %s', n, file);
sig = {'Motor.Motor.M.T', 'Inverter.Inverter.M.T', 'DCDC.DC_DC_Converter.M.T', ...
       'Compressor.Compressor.A.p', 'Compressor.Compressor.B.p', 'Cabin.Cabin_Air_Volume.T_I', ...
       'Compressor.Compressor.omega', 'Blower.Blower.mdot_A', 'Motor_Pump.Motor_Pump.omega', ...
       'Battery_Pump.Battery_Pump.omega', 'Heater.Controlled_Heat_Flow_Rate_Source.S', 'Radiator.Fan.omega', ...
       'Motor.Controlled_Heat_Flow_Rate_Source.S', 'Inverter.Controlled_Heat_Flow_Rate_Source.S', ...
       'DCDC.Controlled_Heat_Flow_Rate_Source.S', 'Battery.Current_Demand.i', 'Cabin.Environment_Air.T', ...
       'Heater.Heater_Coolant_Volume.T_I'};
cellfun(need, sig);

D.t = M.time(:)';
packT = 0; soc = 0;
for k = 1:4
    packT = packT + g(sprintf('Battery.Pack_%d.Thermal_Model.Battery_Thermal_Mass.T', k)) / 4;
    soc   = soc   + g(sprintf('Battery.Pack_%d.Lithium_Cell_1RC.Em_table.SOC', k)) / 4;
end
wallC = [36.4*840, 75.0*500, 30.0*500];                 % same weighting as the COM cabin interior
walls = [g('Cabin.Cabin_Heat_Transfer.Glass_Thermal_Mass.M.T'), ...
         g('Cabin.Cabin_Heat_Transfer.Doors_Thermal_Mass.M.T'), ...
         g('Cabin.Cabin_Heat_Transfer.Roof_Thermal_Mass.M.T')];
cabInt = walls * wallC(:) / sum(wallC);

% states [motT invT dcdcT SOC batT p_in p_out cabIntT cabAirT] in K / Pa (COM order)
D.xref = [g('Motor.Motor.M.T'), g('Inverter.Inverter.M.T'), g('DCDC.DC_DC_Converter.M.T'), soc, packT, ...
          1e6 * g('Compressor.Compressor.A.p'), 1e6 * g('Compressor.Compressor.B.p'), cabInt, ...
          g('Cabin.Cabin_Air_Volume.T_I')]';
% inputs [compNrpm cabAirMfIn motPumpNrpm batPumpNrpm heaterPwr radFanrpm] (COM order)
D.u = [g('Compressor.Compressor.omega'), g('Blower.Blower.mdot_A'), g('Motor_Pump.Motor_Pump.omega'), ...
       g('Battery_Pump.Battery_Pump.omega'), g('Heater.Controlled_Heat_Flow_Rate_Source.S'), ...
       g('Radiator.Fan.omega')]';
D.motQ  = g('Motor.Controlled_Heat_Flow_Rate_Source.S')';
D.invQ  = g('Inverter.Controlled_Heat_Flow_Rate_Source.S')';
D.dcdcQ = g('DCDC.Controlled_Heat_Flow_Rate_Source.S')';
D.batI  = g('Battery.Current_Demand.i')';
D.Tamb_C = (g('Cabin.Environment_Air.T') - 273.15)';
D.clntBatTin_K = (g('Heater.Heater_Coolant_Volume.T_I') + 273.15)';
if any(strcmp(names, 'simlog.Inverter.Coolant_Jacket.B.T'))
    D.clntInvTout_K = g('Inverter.Coolant_Jacket.B.T')';
else
    D.clntInvTout_K = g('Inverter.Coolant_Jacket.A.T')';
end
[tv, iu] = unique(speedTs.Time(:)); dv = speedTs.Data(:);
D.speed = interp1(tv, dv(iu), D.t, 'linear', 0);       % km/h, as in the MPC
end

% =========================================================================
function [X, ood, nfail] = simulate_com(F, packParams, D, mode, TV)
clear get_tuneS thermalLogicAndEstimation
assignin('base', 'SCHED_MODE', mode);
evalin('base', 'clear gamma_log');
N = numel(D.t); dt = 1;
X = NaN(9, N); x = D.xref(:, 1); X(:, 1) = x;
inputInit = struct('envT', D.Tamb_C(1) + 273.15, 'clntInvTout_K', D.clntInvTout_K(1), ...
                   'clntBatTin_K', D.clntBatTin_K(1), 'compN', D.u(1, 1), 'pumpN', D.u(3, 1), ...
                   'prevMode', 0, 'heaterPwr', D.u(5, 1));
est_last = []; nfail = 0;
for k = 1:N - 1
    u = D.u(:, k);
    xpk = pressure_guard(x);
    inputInit.envT = D.Tamb_C(k) + 273.15;
    inputInit.clntInvTout_K = D.clntInvTout_K(k);
    inputInit.clntBatTin_K  = D.clntBatTin_K(k);
    inputInit.compN = u(1); inputInit.pumpN = u(3); inputInit.heaterPwr = u(5);
    heat = struct('dcdcQ', D.dcdcQ(k), 'motQ', D.motQ(k), 'invQ', D.invQ(k), 'speed', D.speed(k), ...
                  'batImeas', max(D.batI(k), 1e-3), 'superHeat', 2);
    tuneS = get_tuneS(xpk, D.Tamb_C(k), D.batI(k), D.speed(k), TV);
    try
        [ctrl, est] = thermalLogicAndEstimation(xpk, inputInit, heat);
        est_last = est;
    catch
        nfail = nfail + 1;                              % CoolProp failed at this operating point
        if isempty(est_last), error('validate:est', 'Property estimation failed at the first step.'); end
        est = est_last;
    end
    est.speed = D.speed(k); est.batImeas = max(D.batI(k), 1e-3);
    est.motQmeas = D.motQ(k); est.invQmeas = D.invQ(k); est.dcdcQmeas = D.dcdcQ(k);
    est.superheatK = 2; est.fanRpm_prev = u(6);
    if isfield(ctrl, 'thermalMode'), inputInit.prevMode = ctrl.thermalMode; end
    p = packParams(est, ctrl, tuneS);
    f = @(xx) full(F(xx, u, p));
    k1 = f(x); k2 = f(x + dt/2*k1); k3 = f(x + dt/2*k2); k4 = f(x + dt*k3);
    xn = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4);            % RK4, one step per second (as in the MPC)
    if ~all(isfinite(xn))
        warning('validate:diverged', '[%s] COM diverged at t = %d s', mode, k); break;
    end
    x = xn; X(:, k + 1) = x;
    if mod(k, 600) == 0, fprintf('   t = %4d s\n', k); end
end
ood = zeros(1, N);
if evalin('base', 'exist(''gamma_log'',''var'')')
    gl = evalin('base', 'gamma_log'); ood(1:size(gl, 1)) = gl(:, 3)';
end
end

function xp = pressure_guard(x)
% same cold-start guard the MPC uses before property estimation
xp = x;
xp(6) = min(max(xp(6), 0.5e5), 20e5);
xp(7) = min(max(xp(7), 0.5e5), 50e5);
if xp(7) < 1.5 * xp(6)
    xp(6) = max(xp(6), 2.0e5);
    xp(7) = max(xp(7), max(xp(6) * 1.5, 6.0e5));
end
end

% =========================================================================
function make_figures(Rt, D, modes, T, outDir)
C = lines(numel(modes));
lbl = @(m) strrep(m, 'interp', 'interpolation (paper)');
t = D.t;
% --- Fig. 2 layout: cabin air, battery, p_in, p_out
fig = figure('Visible', 'off', 'Position', [100 100 1100 800]);
spec = {9, 'Cabin air temperature T_{cair}', '\circC', -273.15, 1; ...
        5, 'Battery temperature T_b',        '\circC', -273.15, 1; ...
        6, 'Low-side pressure p_{in}',       'bar',     0, 1e-5; ...
        7, 'High-side pressure p_{out}',     'bar',     0, 1e-5};
for s = 1:4
    subplot(2, 2, s); hold on; grid on;
    i = spec{s, 1}; off = spec{s, 4}; sc = spec{s, 5};
    plot(t, (D.xref(i, :) + off) * sc, 'k--', 'LineWidth', 1.4, 'DisplayName', sprintf('Ref %d\\circC', T));
    for m = 1:numel(modes)
        X = Rt.(matlab.lang.makeValidName(modes{m})).X;
        plot(t, (X(i, :) + off) * sc, 'Color', C(m, :), 'LineWidth', 1.2, 'DisplayName', ['COM, ' lbl(modes{m})]);
    end
    xlabel('Time (s)'); ylabel(sprintf('%s (%s)', spec{s, 2}(1:find(spec{s, 2} == ' ', 1, 'last') - 1), spec{s, 3}));
    title(spec{s, 2}); if s == 1, legend('Location', 'southeast', 'FontSize', 8); end
end
sgtitle(sprintf('COM vs. high-fidelity reference at %d\\circC (open loop, rule-based inputs)', T));
save_fig(fig, outDir, sprintf('fig2_T%d', abs(T)));
% --- remaining thermal states
fig = figure('Visible', 'off', 'Position', [100 100 1300 380]);
spec2 = {1, 'Motor T_{mot}'; 2, 'Inverter T_{inv}'; 3, 'DC-DC T_{dcdc}'};
for s = 1:3
    subplot(1, 3, s); hold on; grid on; i = spec2{s, 1};
    plot(t, D.xref(i, :) - 273.15, 'k--', 'LineWidth', 1.4, 'DisplayName', 'Ref');
    for m = 1:numel(modes)
        X = Rt.(matlab.lang.makeValidName(modes{m})).X;
        plot(t, X(i, :) - 273.15, 'Color', C(m, :), 'LineWidth', 1.2, 'DisplayName', lbl(modes{m}));
    end
    xlabel('Time (s)'); ylabel('\circC'); title(spec2{s, 2}); if s == 1, legend('Location', 'southeast'); end
end
save_fig(fig, outDir, sprintf('components_T%d', abs(T)));
end

function save_fig(fig, d, name)
savefig(fig, fullfile(d, [name '.fig']));
exportgraphics(fig, fullfile(d, [name '.png']), 'Resolution', 200);
try, matlab2tikz(fullfile(d, [name '.tex']), 'figurehandle', fig, 'showInfo', false); catch, end
close(fig);
end
