function export_com_testcases(nCases)
% EXPORT_COM_TESTCASES  Evaluate the COM (dynamics_mexfreeNew.m) at many test points and save
% inputs + outputs, so the PyTorch port (python/com_torch.py) can be checked against MATLAB exactly.
%
%   export_com_testcases          % 400 test points
%   export_com_testcases(1000)
%
% Writes  python/com_testcases.mat. Takes a few seconds, no Simulink run needed.
% Test points: states, properties and supervisor commands from Data/tuning_data_{5,7,10}.mat, with
%   - random gamma (p1..p11), log-uniform in [0.1, 10]
%   - for half of the points, states perturbed randomly (so the test is not only on the stored trajectory)
%   - all control inputs randomised within their physical ranges.

    if nargin < 1, nCases = 400; end
    repoRoot = fileparts(fileparts(mfilename('fullpath')));       % this file is in "Source Files"
    addpath(fullfile(repoRoot, 'Source Files'));
    rng(1);

    [~, U_ordering] = local_orderings();
    estNames  = U_ordering.estimatedParams;
    ctrlNames = U_ordering.controlOut;
    uNames    = U_ordering.inputSignal;

    X = zeros(nCases, 9); U = zeros(nCases, numel(uNames));
    EST = zeros(nCases, numel(estNames)); CTRL = zeros(nCases, numel(ctrlNames));
    G = zeros(nCases, 11); XDOT = zeros(nCases, 9); Y = []; J = zeros(nCases, 1);

    temps = [5 7 10];
    for n = 1:nCases
        T = temps(mod(n-1, 3) + 1);
        if n <= 3 || ~exist('cache', 'var') || ~isfield(cache, sprintf('T%d', T))
            cache.(sprintf('T%d', T)) = load(fullfile(repoRoot, 'Data', sprintf('tuning_data_%d.mat', T)));
        end
        d = cache.(sprintf('T%d', T));
        k = randi([2, 3599]);

        sp = d.sim_params;    if iscell(sp), s = sp{k}; else, s = sp(k); end
        co = d.controlOutCell; if iscell(co), c = co{k}; else, c = co(k); end
        r  = d.resampled;

        % ---- states ----
        x = d.States(:, k);
        if mod(n, 2) == 0
            x([1 2 3 5 8 9]) = x([1 2 3 5 8 9]) + 5*randn(6, 1);        % +-5 K
            x(4) = min(max(x(4) + 0.05*randn, 0.05), 1);                 % SOC
            x(6:7) = x(6:7) .* exp(0.2*randn(2, 1));                     % pressures +-20 %
        end

        % ---- estimated parameters (struct, as in the NMPC) ----
        est = struct();
        for i = 1:numel(estNames)
            nm = estNames{i};
            if isfield(s, nm), est.(nm) = double(s.(nm)); end
        end
        est.envT        = double(s.envT) + 273.15 * (s.envT < 100);     % stored in degC in these files
        est.batImeas    = double(r.batImeas(k));
        est.speed       = 0;
        est.fanRpm_prev = 0;
        for i = 1:numel(estNames)                                        % anything still missing -> 0
            if ~isfield(est, estNames{i}), est.(estNames{i}) = 0; end
        end

        % ---- supervisor commands ----
        ctrl = struct();
        for i = 1:numel(ctrlNames)
            nm = ctrlNames{i};
            if isfield(c, nm), ctrl.(nm) = double(c.(nm)); else, ctrl.(nm) = 0; end
        end

        % ---- control inputs: random within physical ranges ----
        u = struct('compNrpm', 5000*rand, 'cabAirMfIn', 0.2*rand, 'motPumpNrpm', 1200*rand, ...
                   'batPumpNrpm', 1200*rand, 'heaterPwr', 4000*rand, 'radFanrpm', 1000*rand);

        gamma = exp(log(0.1) + (log(10) - log(0.1))*rand(11, 1));

        p = struct('est', est, 'ctrl', ctrl, 'tuneParam', gamma);
        [xdot, y, Jn] = dynamics_mexfreeNew(x, u, p);

        X(n, :) = x'; G(n, :) = gamma'; XDOT(n, :) = xdot'; J(n) = Jn;
        if isempty(Y), Y = zeros(nCases, numel(y)); end
        Y(n, :) = y';
        for i = 1:numel(uNames),    U(n, i)    = u.(uNames{i});    end
        for i = 1:numel(estNames),  EST(n, i)  = est.(estNames{i}); end
        for i = 1:numel(ctrlNames), CTRL(n, i) = ctrl.(ctrlNames{i}); end
    end

    out = fullfile(repoRoot, 'python', 'com_testcases.mat');
    save(out, 'X', 'U', 'EST', 'CTRL', 'G', 'XDOT', 'Y', 'J', 'estNames', 'ctrlNames', 'uNames');
    fprintf('[export_com_testcases] %d test points saved to %s\n', nCases, out);
end

function [F, U_ordering] = local_orderings()
% Same orderings as create_dynamics_nmpcNew.m, without building the CasADi function.
    F = [];
    U_ordering.inputSignal = {'compNrpm', 'cabAirMfIn', 'motPumpNrpm', 'batPumpNrpm', 'heaterPwr', 'radFanrpm'};
    U_ordering.controlOut = {'cmdHeater', 'cmdParallelSerial', 'cmdBlendAir', 'cmdExvEvap', ...
                             'cmdExvChiller', 'thermalMode', 'cmdExvHeatpump'};
    U_ordering.estimatedParams = { ...
        'envT', 'clntCp','clntK','clntRho','clntMu', ...
        'cabAirCp','cabAirK','cabAirRho','cabAirMu', ...
        'innerCondRefCp','innerCondRefK','innerCondRefRho','innerCondRefMu', ...
        'condEvapRefCp','condEvapRefK','condEvapRefRho','condEvapRefMu', ...
        'condEvapAirCp','condEvapAirK','condEvapAirRho','condEvapAirMu', ...
        'evapRefCp','evapRefK','evapRefRho','evapRefMu', ...
        'chilRefCp','chilRefK','chilRefRho','chilRefMu', ...
        'speed', 'compHin','compHoutS','compVnominalIn', ...
        'refTsatHighPK','refCpVapHighP', ...
        'refHsatLiqHighP','refTsatLiqHighPK','refTsatLiqLowPK', ...
        'motQmeas','invQmeas','dcdcQmeas', ...
        'evapVoidFrac','condVoidFrac', 'evapVol_m3','condVol_m3', ...
        'evapWallThermalMass_J_K','condWallThermalMass_J_K', ...
        'dTesat_dp1','d_rhoh_l_dp1','d_rhoh_g_dp1', ...
        'dTcsat_dp2','d_rhoh_l_dp2','d_rhoh_g_dp2', ...
        'batImeas','refCpLiqHighP', 'fanRpm_prev'};
end
