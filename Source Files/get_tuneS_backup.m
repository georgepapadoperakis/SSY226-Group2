function tuneS = get_tuneS(x_meas, Tamb_C, batI, speed, TV)
%GET_TUNES  Parameter scheduler for the NMPC: drop-in replacement for prepParaVec.
%
%   tuneS = get_tuneS(x_meas, Tamb_C, batI, speed, TV)
%
%   Returns the same struct as prepParaVec (fields p1..p11). Which scheduler is
%   used is set in the base workspace BEFORE the simulation starts:
%
%       SCHED_MODE = 'interp';     % original linear interpolation (prepParaVec)
%       SCHED_MODE = 'george';     % neural network from nn_george.mat
%       SCHED_MODE = 'partner1';   % neural network from nn_partner1.mat ...
%
%   Out-of-distribution fallback: if the current operating point lies outside
%   the network's training envelope (env_lo/env_hi in the nn_*.mat file), or the
%   network returns non-finite values, the interpolated parameters are used.
%
%   Every call is logged to the base workspace variable gamma_log:
%       [step, Tamb_C, ood_flag, p1..p11 (used), p1..p11 (interpolation)]
%
%   Call  clear get_tuneS  before each new simulation (the runner does this).
%
%   NN file format (nn_<name>.mat), written by export_nn_for_matlab.py:
%     features      cellstr, input names in order (see build_features below)
%     x_mean,x_std  input normalisation
%     n_layers, W1,b1 .. Wn,bn   dense layers, W is (out x in) as in PyTorch
%     act           hidden activation: 'tanh' | 'relu' | 'sigmoid' | 'gelu' | 'softplus'
%     y_mean,y_std  (optional) output de-normalisation
%     out_transform (optional) 'none' (default) | 'exp'  (network predicts log p)
%     p_lo,p_hi     (optional) clamp of p1..p11 to a plausible range
%     env_lo,env_hi (optional) feature envelope for the OOD fallback
%#ok<*AGROW>

persistent mode net glog k

if isempty(mode)
    mode = 'interp';
    try
        if evalin('base', 'exist(''SCHED_MODE'',''var'')')
            mode = char(evalin('base', 'SCHED_MODE'));
        end
    catch
    end
    glog = zeros(0, 25);
    k = 0;
    net = [];
    if ~strcmpi(mode, 'interp')
        f = ['nn_' mode '.mat'];
        if ~isfile(f) && isempty(which(f))
            error('get_tuneS:noNet', 'SCHED_MODE=''%s'' but %s was not found on the path.', mode, f);
        end
        net = load(f);
        net.features = cellstr(net.features);
        net.act = char(net.act);
        if ~isfield(net, 'out_transform'), net.out_transform = 'none'; end
        net.out_transform = char(net.out_transform);
        fprintf('[get_tuneS] Scheduler: neural network %s (%d inputs, %d layers, act=%s)\n', ...
                f, numel(net.features), double(net.n_layers), net.act);
    else
        fprintf('[get_tuneS] Scheduler: linear interpolation (prepParaVec)\n');
    end
end

k = k + 1;
p_interp = struct2vec(prepParaVec(x_meas, Tamb_C, TV));
ood = 0;

if strcmpi(mode, 'interp')
    p = p_interp;
else
    feat = build_features(net.features, x_meas, Tamb_C, batI(1), speed(1));
    if isfield(net, 'env_lo') && any(feat < net.env_lo(:) | feat > net.env_hi(:))
        p = p_interp; ood = 1;
    else
        p = nn_forward(net, feat);
        if ~all(isfinite(p)), p = p_interp; ood = 1; end
    end
end

glog(end+1, :) = [k, Tamb_C, ood, p(:).', p_interp(:).'];
assignin('base', 'gamma_log', glog);

for i = 1:11
    tuneS.(sprintf('p%d', i)) = p(i);
end
end

% -------------------------------------------------------------------------
function v = struct2vec(s)
v = zeros(11, 1);
for i = 1:11
    v(i) = s.(sprintf('p%d', i));
end
end

% -------------------------------------------------------------------------
function f = build_features(names, x, Tamb_C, batI, speed)
% Base signals in MPC units: temperatures K, pressures Pa, SOC [-], current A,
% speed as passed to the MPC. Suffixes convert units:
%   _C   : temperature in degC      (e.g. 'batT_C', 'cabAirT_C')
%   _bar : pressure in bar          (e.g. 'p_in_bar')
%   _MPa : pressure in MPa
base = struct('motT', x(1), 'invT', x(2), 'dcdcT', x(3), 'SOC', x(4), 'batT', x(5), ...
              'p_in', x(6), 'p_out', x(7), 'cabIntT', x(8), 'cabAirT', x(9), ...
              'Tamb_K', Tamb_C + 273.15, 'batI', batI, 'speed', speed);
f = zeros(numel(names), 1);
for i = 1:numel(names)
    nm = names{i};
    if strcmp(nm, 'Tamb_C')
        f(i) = Tamb_C;
    elseif isfield(base, nm)
        f(i) = base.(nm);
    elseif endsWith(nm, '_C') && isfield(base, nm(1:end-2))
        f(i) = base.(nm(1:end-2)) - 273.15;
    elseif endsWith(nm, '_bar') && isfield(base, nm(1:end-4))
        f(i) = base.(nm(1:end-4)) / 1e5;
    elseif endsWith(nm, '_MPa') && isfield(base, nm(1:end-4))
        f(i) = base.(nm(1:end-4)) / 1e6;
    else
        error('get_tuneS:unknownFeature', 'Unknown network input ''%s''.', nm);
    end
end
end

% -------------------------------------------------------------------------
function p = nn_forward(net, f)
z = (f(:) - net.x_mean(:)) ./ net.x_std(:);
L = double(net.n_layers);
for i = 1:L
    W = net.(sprintf('W%d', i));
    b = net.(sprintf('b%d', i));
    z = W * z + b(:);
    if i < L
        switch lower(net.act)
            case 'tanh',     z = tanh(z);
            case 'relu',     z = max(z, 0);
            case 'sigmoid',  z = 1 ./ (1 + exp(-z));
            case 'softplus', z = log(1 + exp(z));
            case 'gelu',     z = 0.5 * z .* (1 + erf(z / sqrt(2)));
            otherwise, error('get_tuneS:act', 'Unknown activation %s', net.act);
        end
    end
end
if isfield(net, 'y_mean'), z = z .* net.y_std(:) + net.y_mean(:); end
if strcmpi(net.out_transform, 'exp'), z = exp(z); end
p = z(:);
if isfield(net, 'p_lo'), p = max(p, net.p_lo(:)); end
if isfield(net, 'p_hi'), p = min(p, net.p_hi(:)); end
end
