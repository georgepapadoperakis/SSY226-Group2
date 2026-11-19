function cycFile = make_drive_cycles(eps_dir)
%MAKE_DRIVE_CYCLES  Build DriveCycles_SSY226.mat: every cycle in drive_cycle_list as a Signal Editor scenario.
%
%   make_drive_cycles                 % EPA files looked for in Data/drive_cycles/
%   make_drive_cycles('C:\...\cycles')
%
%   Every new cycle is repeated (and cut) to the same 3600 s as the original extended cycle,
%   so every run is a one-hour cold start like the existing data.
%
%   Output file (repo root): DriveCycles_SSY226.mat, one variable per Signal Editor scenario,
%   named by the cycle handle (see drive_cycle_list):
%     WLTCHW      original WLTC + 130 km/h highway cycle (copied unchanged from the model's file)
%     NEDC        4 x ECE-15 + EUDC (1180 s), built from the regulation's segment table
%     UDDS        EPA urban cycle (1369 s), from uddscol.txt
%     US06        EPA aggressive cycle (600 s), from us06col.txt   -> stress test
%     DriveCycle, ZeroSpeed   the model's original scenarios, kept unchanged
%   Each variable is a Simulink.SimulationData.Dataset with one timeseries [km/h], same signal name
%   as the original, so the Signal Editor block can switch between them with 'ActiveScenario'.
%
%   UDDS / US06 speed files: download from the EPA "Dynamometer Drive Schedules" page
%   (uddscol.txt, us06col.txt: two header lines, then  time [s]  speed [mph]) and put them in
%   Data/drive_cycles/ (already there in the repo).

    repoRoot = fileparts(fileparts(mfilename('fullpath')));        % this file lives in "Source Files"
    if nargin < 1, eps_dir = fullfile(repoRoot, 'Data', 'drive_cycles'); end
    T_TOTAL = 3600;                                                   % [s] same length as the original cycle

    orig = load(fullfile(repoRoot, 'ElectricVehicleThermalManagementWithHeatPumpVehicleSpeed.mat'));
    ref  = orig.DriveCycle{1};                                        % timeseries, km/h
    names   = orig.DriveCycle.getElementNames;                       % Signal Editor matches signals by this name
    sigName = names{1};
    if isempty(sigName), sigName = ref.Name; end
    fprintf('[cycles] original DriveCycle: signal ''%s'', %.0f s, max %.0f km/h\n', ...
            sigName, ref.Time(end), max(ref.Data));

    S = struct();
    S.WLTCHW     = orig.DriveCycle;
    S.DriveCycle = orig.DriveCycle;
    S.ZeroSpeed  = orig.ZeroSpeed;

    cycles = struct('name', {}, 't', {}, 'v', {});
    [t, v] = nedc();                                     cycles(end+1) = struct('name', 'NEDC', 't', t, 'v', v);
    [t, v] = epa_file(fullfile(eps_dir, 'uddscol.txt'));  cycles(end+1) = struct('name', 'UDDS', 't', t, 'v', v);
    [t, v] = epa_file(fullfile(eps_dir, 'us06col.txt'));  cycles(end+1) = struct('name', 'US06', 't', t, 'v', v);

    missing = setdiff({drive_cycle_list().handle}, [{'WLTCHW'}, {cycles.name}]);
    if ~isempty(missing)
        error('make_drive_cycles:notBuilt', 'drive_cycle_list has cycles this file does not build: %s', strjoin(missing, ', '));
    end

    figure('Name', 'Drive cycles (3600 s)'); tl = tiledlayout(numel(cycles) + 1, 1, 'TileSpacing', 'compact');
    nexttile; plot(ref.Time, ref.Data); title('WLTCHW (original)'); ylabel('km/h'); grid on

    for c = cycles
        [t, v] = repeat_to(c.t, c.v, T_TOTAL);
        ts = timeseries(v, t, 'Name', sigName);
        ts.DataInfo.Units = ref.DataInfo.Units;
        ts.DataInfo.Interpolation = ref.DataInfo.Interpolation;
        ds = Simulink.SimulationData.Dataset;
        ds = ds.addElement(ts, sigName);
        S.(c.name) = ds;
        fprintf('[cycles] %-5s one cycle %4.0f s, %5.2f km | repeated: %4.0f s, %5.1f km, mean %4.1f km/h, max %5.1f km/h, idle %2.0f %%\n', ...
                c.name, c.t(end), trapz(c.t, c.v) / 3600, t(end), trapz(t, v) / 3600, mean(v), max(v), 100 * mean(v < 0.5));
        nexttile; plot(t, v); title(c.name); ylabel('km/h'); grid on
    end
    xlabel(tl, 'time [s]');

    cycFile = fullfile(repoRoot, 'DriveCycles_SSY226.mat');
    save(cycFile, '-struct', 'S');
    fprintf('[cycles] saved %s with scenarios: %s\n', cycFile, strjoin(fieldnames(S), ', '));
end


function [t, v] = repeat_to(t1, v1, T)
% Repeat one cycle back to back until T seconds, then cut at T (1-s grid).
    t1 = t1(:) - t1(1); v1 = v1(:);
    L = t1(end);
    n = ceil(T / L);
    t = []; v = [];
    for k = 0:n-1
        if k == 0, idx = 1:numel(t1); else, idx = 2:numel(t1); end   % no duplicate time at the joins
        t = [t; t1(idx) + k * L]; %#ok<AGROW>
        v = [v; v1(idx)];         %#ok<AGROW>
    end
    tg = (0:T)';
    v  = interp1(t, v, tg, 'linear');
    t  = tg;
end


function [t, v] = epa_file(f)
% EPA schedule file: 2 header lines, then "time[s]  speed[mph]". Returns km/h.
    if ~isfile(f)
        error('make_drive_cycles:missing', ['Speed file not found: %s\n' ...
              'Download it from the EPA "Dynamometer Drive Schedules" page into Data/drive_cycles/.'], f);
    end
    fid = fopen(f, 'r');
    closer = onCleanup(@() fclose(fid));
    C = textscan(fid, '%f %f', 'HeaderLines', 2);
    t = C{1};
    v = C{2} * 1.609344;                                              % mph -> km/h
    ok = isfinite(t) & isfinite(v);
    t = t(ok); v = v(ok);
    if numel(t) < 100
        error('make_drive_cycles:badFile', 'Could not read %s (expected 2 header lines, then time and mph).', f);
    end
end


function [t, v] = nedc()
% NEDC from the segment table (UNECE R83): 4 x ECE-15 (195 s each) + EUDC (400 s) = 1180 s.
% Each row: [duration s, speed at end of segment km/h]; speed is linear within a segment.
    ece = [11 0;  4 15;  8 15;  5 0;  21 0;                          % idle, 0-15, cruise 15, 15-0, idle
           5 15;  2 15;  5 32;  24 32;  11 0;  21 0;                 % 0-15, gear change, 15-32, cruise, 32-0, idle
           5 15;  2 15;  9 35;  2 35;  8 50;  12 50;                 % 0-15, gc, 15-35, gc, 35-50, cruise 50
           8 35;  13 35;  12 0;  7 0];                               % 50-35, cruise 35, 35-0, idle
    eudc = [20 0;  5 15;  2 15;  9 35;  2 35;  8 50;  2 50;  13 70;  % idle, 0-15, gc, 15-35, gc, 35-50, gc, 50-70
            50 70;  8 50;  69 50;  13 70;  50 70;                    % cruise 70, 70-50, cruise 50, 50-70, cruise 70
            35 100;  30 100;  20 120;  10 120;                       % 70-100, cruise 100, 100-120, cruise 120
            16 80;  8 50;  10 0;  20 0];                             % 120-80, 80-50, 50-0, idle
    seg = [repmat(ece, 4, 1); eudc];
    t = [0; cumsum(seg(:, 1))];
    v = [0; seg(:, 2)];
    assert(t(end) == 1180, 'NEDC table should total 1180 s, got %g', t(end));
    tg = (0:1180)';
    v = interp1(t, v, tg, 'linear');
    t = tg;
end
