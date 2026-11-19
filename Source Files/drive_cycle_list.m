function C = drive_cycle_list(handle)
%DRIVE_CYCLE_LIST  The drive cycles the project can simulate, in one place.
%
%   C = drive_cycle_list          % all cycles (struct array)
%   c = drive_cycle_list('UDDS')  % one cycle; also accepts 'DriveCycle' (= 'WLTCHW')
%
%   Fields:
%     handle       short name used everywhere: menu, Results/run_<time>_<handle>/,
%                  training_data_<ctrl>_T<|T|>_<handle>.mat, trainingData.cycle, Python loader
%     scenario     Signal Editor scenario name in DriveCycles_SSY226.mat (equal to the handle)
%     description  one line for the menu
%
%   'WLTCHW' is the original project cycle (WLTC followed by ~130 km/h highway, 3600 s): the one the
%   paper and all earlier data use. Exports from before 18 Nov have no handle in their file name and
%   are treated as WLTCHW.
%
%   To add a cycle: add a row here and build it in make_drive_cycles.m.

    rows = {
        'WLTCHW', 'Original: WLTC + 130 km/h highway (paper, all earlier data)'
        'NEDC',   'NEDC: 4 x urban ECE-15 + extra-urban EUDC, smooth and modal'
        'UDDS',   'UDDS (EPA FTP-72): city stop-and-go, much idling'
        'US06',   'US06 (EPA): aggressive, hard accelerations - stress test'
    };
    C = struct('handle', rows(:, 1), 'scenario', rows(:, 1), 'description', rows(:, 2));

    if nargin == 0
        return
    end
    handle = char(handle);
    if strcmpi(handle, 'DriveCycle'), handle = 'WLTCHW'; end
    k = find(strcmpi({C.handle}, handle), 1);
    if isempty(k)
        error('drive_cycle_list:unknown', 'Unknown drive cycle ''%s''. Known: %s', ...
              handle, strjoin({C.handle}, ', '));
    end
    C = C(k);
end
