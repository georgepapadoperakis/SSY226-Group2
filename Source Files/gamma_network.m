function tune = gamma_network(x, Tamb_C, Uplan, motQ, invQ, batI)
% GAMMA_NETWORK  COM tuning parameters p1..p15 from the trained gamma network (replaces prepParaVec).
%
%   tune = gamma_network(x, Tamb_C, Uplan, motQ, invQ, batI)
%     x       9x1 measured state [Tmot Tinv Tdcdc SOC Tb p_in p_out Tcint Tcair] (K, Pa)
%     Tamb_C  ambient temperature (degC)
%     Uplan   6xN input plan over the horizon [compNrpm; cabAirMfIn; motPumpNrpm; batPumpNrpm; heaterPwr; radFanrpm]
%             (in the NMPC: the previous optimal plan shifted by one step). Column 1 = input for the coming second.
%     motQ, invQ, batI   motor heat (W), inverter heat (W), battery current (A) now
%     tune    struct of tuning parameters, ready for packParams:
%             COM_FIXES = true : p1..p15 for the fixed COM            (python/models/gamma15_fixed_com.mat)
%             COM_FIXES = false: p1..p11, p12 = 0 for the original COM (python/models/gamma11_original_com.mat)
%
%   gamma_network()   self-test: compares with the PyTorch network on 193 stored test cases.
%
% Network: 25 inputs -> 64 tanh -> 64 tanh -> gamma1..11 in [0.1, 10] (+ for the fixed COM: gamma12 in [0, 1.5],
% gamma13..15 in [0.1, 10]). Weights exported by python/export_networks_to_matlab.py.
% The inputs are built exactly as features() in python/partB.py.

    persistent net netName
    comFixes = true;
    try, comFixes = logical(evalin('base', 'COM_FIXES')); catch, end
    if comFixes, name = 'gamma15_fixed_com.mat'; else, name = 'gamma11_original_com.mat'; end
    if isempty(net) || ~strcmp(netName, name)                       % (re)load when the COM switch changes
        repoRoot = fileparts(fileparts(mfilename('fullpath')));      % this file is in "Source Files"
        net = load(fullfile(repoRoot, 'python', 'models', name));
        netName = name;
    end

    if nargin == 0
        self_test(net);
        tune = [];
        return
    end

    g = forward(net, features(net, x, Tamb_C, Uplan, motQ, invQ, batI));
    tune = struct();
    for i = 1:numel(g)
        tune.(sprintf('p%d', i)) = g(i);
    end
    if numel(g) == 11
        tune.p12 = 0;                                  % original COM (no suction-density correction)
    end
end

function z = features(net, x, Tamb_C, Uplan, motQ, invQ, batI)
    x = x(:);
    temps = (x([1 2 3 5 8 9]) - 273.15) / 50;          % motor, inverter, DC-DC, battery, cabin interior, cabin air
    us = net.u_scale(:);
    u0 = Uplan(:, 1) ./ us;                            % input for the coming second
    um = mean(Uplan, 2) ./ us;                         % mean over the 30-s preview
    q  = [motQ; invQ; batI] ./ net.q_scale(:);
    z = [Tamb_C/20; temps; x(4); x(6:7)/1e6; u0; um; q];
end

function g = forward(net, z)
    zn = (z(:) - net.mu(:)) ./ net.sd(:);
    h1 = tanh(net.W1*zn + net.b1(:));
    h2 = tanh(net.W2*h1 + net.b2(:));
    sig = @(v) 1 ./ (1 + exp(-v));
    g = exp(net.log_lo + sig(net.W3*h2 + net.b3(:)) * (net.log_hi - net.log_lo));
    if isfield(net, 'W12')                             % fixed-COM network: gamma12 and gamma13..15
        g12 = net.g12_max * sig(net.W12*h2 + net.b12(:));
        g13 = exp(net.log_lo + sig(net.Wh*h2 + net.bh(:)) * (net.log_hi - net.log_lo));
        g = [g; g12; g13];
    end
end

function self_test(net)
    n = size(net.test_x, 1);
    errG = 0; errZ = 0;
    for i = 1:n
        Uplan = squeeze(net.test_Uplan(i, :, :)).';    % 6 x 30
        z = features(net, net.test_x(i, :), net.test_Tamb(i), Uplan, net.test_motQ(i), net.test_invQ(i), net.test_batI(i));
        g = forward(net, z);
        errZ = max(errZ, max(abs(z(:).' - net.test_features(i, :))));
        errG = max(errG, max(abs(g(:).' - net.test_gamma(i, :)) ./ abs(net.test_gamma(i, :))));
    end
    fprintf('[gamma_network] %d test cases: max |input diff| %.2e, max relative gamma diff %.2e  ->  %s\n', ...
            n, errZ, errG, ternary(errG < 1e-9, 'PASS', 'CHECK'));
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end
