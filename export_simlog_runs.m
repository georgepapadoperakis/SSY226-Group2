function export_simlog_runs(files, dt, outDir)
% EXPORT_SIMLOG_RUNS  Flatten Simscape logs into plain .mat files Python can read.
%
%   export_simlog_runs                                  % all simlog_rule_*.mat in this folder, 1 s step
%   export_simlog_runs({'simlog_rule_T5.mat'}, 1, 'Results/exported')
%
% For every input file it writes <name>_export.mat containing:
%   time   [N x 1]  common time vector (s)
%   data   [N x M]  every logged signal, resampled onto time
%   names  {M x 1}  signal paths, e.g. 'simlog.Battery.Pack.T'
%   units  {M x 1}  units of each signal
% plus <name>_signals.txt listing all signal names.

if nargin < 1 || isempty(files)
    d = dir('simlog_rule_*.mat');
    files = {d.name};
end
if ischar(files) || isstring(files), files = cellstr(files); end
if nargin < 2 || isempty(dt), dt = 1; end
if nargin < 3 || isempty(outDir), outDir = pwd; end
if ~exist(outDir, 'dir'), mkdir(outDir); end
if isempty(files), error('No simlog_rule_*.mat files found in %s', pwd); end

for f = 1:numel(files)
    inFile = files{f};
    fprintf('\nLoading %s ...\n', inFile);
    S = load(inFile);
    acc = struct('names', {{}}, 'units', {{}}, 't', {{}}, 'v', {{}});
    vars = fieldnames(S);
    for k = 1:numel(vars)
        fprintf('  variable %-20s %s\n', vars{k}, class(S.(vars{k})));
        acc = collectVar(S.(vars{k}), vars{k}, acc);
    end
    clear S
    if isempty(acc.names)
        warning('No time signals found in %s, skipped.', inFile);
        continue
    end

    t0 = max(cellfun(@(t) t(1),   acc.t));
    t1 = min(cellfun(@(t) t(end), acc.t));
    time = (t0:dt:t1)';
    data = zeros(numel(time), numel(acc.names));
    for i = 1:numel(acc.names)
        data(:, i) = interp1(acc.t{i}, acc.v{i}, time, 'linear', 'extrap');
    end
    names = acc.names(:);
    units = acc.units(:);

    [~, base] = fileparts(inFile);
    outFile = fullfile(outDir, [base '_export.mat']);
    save(outFile, 'time', 'data', 'names', 'units', '-v7');

    sigFile = fullfile(outDir, [base '_signals.txt']);
    fid = fopen(sigFile, 'w');
    for i = 1:numel(names)
        fprintf(fid, '%s\t[%s]\n', names{i}, units{i});
    end
    fclose(fid);

    fprintf('  -> %s  (%d samples, %d signals, %.0f..%.0f s)\n', ...
        outFile, numel(time), numel(names), t0, t1);
    fprintf('  -> %s (signal list)\n', sigFile);
end
end

% -------------------------------------------------------------------------
function acc = collectVar(v, prefix, acc)
if isa(v, 'simscape.logging.Node')
    acc = walkNode(v, prefix, acc);

elseif isa(v, 'Simulink.SimulationOutput')
    n = v.who;
    for k = 1:numel(n)
        if ~strcmp(n{k}, 'tout')
            acc = collectVar(v.get(n{k}), n{k}, acc);
        end
    end

elseif isa(v, 'Simulink.SimulationData.Dataset')
    for k = 1:v.numElements
        el = v.getElement(k);
        n = el.Name;
        if isempty(n), n = sprintf('signal%d', k); end
        if isprop(el, 'Values')
            acc = collectVar(el.Values, [prefix '.' n], acc);
        end
    end

elseif isa(v, 'timeseries')
    for k = 1:numel(v)
        n = v(k).Name;
        if isempty(n), n = sprintf('ts%d', k); end
        acc = addSeries(acc, [prefix '.' n], v(k).Time, v(k).Data, '');
    end

elseif isstruct(v) && isscalar(v)
    f = fieldnames(v);
    for k = 1:numel(f)
        acc = collectVar(v.(f{k}), [prefix '.' f{k}], acc);
    end
end
end

% -------------------------------------------------------------------------
function acc = walkNode(node, path, acc)
% Recursively walk a Simscape logging tree and collect every series.
try
    s = node.series;
    if ~isempty(s) && s.points > 1
        acc = addSeries(acc, path, s.time, values(s), s.unit);
    end
catch
end
try
    ids = childIds(node);
catch
    ids = {};
end
for k = 1:numel(ids)
    try
        acc = walkNode(node.(ids{k}), [path '.' ids{k}], acc);
    catch
    end
end
end

% -------------------------------------------------------------------------
function acc = addSeries(acc, name, t, v, unit)
t = double(t(:));
if numel(t) < 2 || ~isnumeric(v) && ~islogical(v), return; end
v = double(squeeze(v));
if size(v, 1) ~= numel(t)
    if size(v, 2) == numel(t)
        v = v.';
    else
        v = reshape(v, [], numel(t)).';
    end
end
[t, ia] = unique(t);
v = v(ia, :);
for c = 1:size(v, 2)
    n = name;
    if size(v, 2) > 1, n = sprintf('%s(%d)', name, c); end
    acc.names{end+1} = n;
    acc.units{end+1} = char(unit);
    acc.t{end+1} = t;
    acc.v{end+1} = real(v(:, c));
end
end