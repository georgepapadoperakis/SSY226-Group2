function make_table1(folder)
%MAKE_TABLE1  Paper-style Table 1 (Lokur & Murgovski) from a saved validation run.
%
%   make_table1                                           % newest Results\com_validation_* folder
%   make_table1('Results\com_validation_2026_11_17_10_21')
%
%   Reads com_validation.mat written by validate_com_fig2 (no re-simulation) and
%   writes table1_paper_style.txt / .csv into the same folder: for every ambient
%   temperature and state, the paper's RMSE/MAE next to ours for every scheduler,
%   plus each NN's RMSE change relative to our interpolation (negative = NN better).

if nargin < 1 || isempty(folder)
    d = dir(fullfile(pwd, 'Results', 'com_validation_*'));
    d = d([d.isdir]);
    if isempty(d), error('make_table1:none', 'No Results\\com_validation_* folder found.'); end
    [~, i] = max([d.datenum]);
    folder = fullfile(d(i).folder, d(i).name);
end
S = load(fullfile(folder, 'com_validation.mat'), 'Tab', 'modes', 'temps');
Tab = S.Tab; modes = cellstr(S.modes); temps = S.temps;

% Table 1 of the paper (RMSE, MAE in degC; interpolation scheduler)
paper = struct('T10', [1.60 1.35; 1.50 1.26; 2.12 1.61; 1.61 1.36; 1.04 0.58], ...
               'T7',  [1.70 1.46; 1.53 1.33; 2.00 1.62; 1.93 1.77; 1.45 1.14], ...
               'T5',  [1.43 1.18; 1.29 1.05; 1.83 1.30; 1.53 1.33; 1.01 0.81]);
stateNames = {'T_mot', 'T_inv', 'T_dcdc', 'T_b', 'T_cair', 'p_in', 'p_out'};
nn = setdiff(modes, {'interp'}, 'stable');

fid = fopen(fullfile(folder, 'table1_paper_style.txt'), 'w');
if fid < 0, error('make_table1:open', 'Cannot write to %s', folder); end
cleanup = onCleanup(@() fclose(fid));

out(fid, '\nTable 1 replication: RMSE / MAE of the COM vs. Simscape over the full drive\n');
out(fid, '(temperatures in degC, pressures in bar; the paper reports temperatures only)\n');
wide = {};
for T = temps(:).'
    tg = sprintf('T%d', abs(T));
    tag = ''; if T == -7, tag = ' (unseen)'; end
    out(fid, '\n=== %d degC%s ===\n', T, tag);
    hdr = sprintf('%-8s %13s', 'State', 'paper');
    for m = 1:numel(modes), hdr = [hdr sprintf(' %15s', ['ours-' modes{m}])]; end %#ok<AGROW>
    for m = 1:numel(nn),    hdr = [hdr sprintf(' %17s', [nn{m} ' vs interp'])]; end %#ok<AGROW>
    out(fid, '%s\n', hdr);
    for s = 1:numel(stateNames)
        pr = [NaN NaN];
        if s <= 5 && isfield(paper, tg), pr = paper.(tg)(s, :); end
        txt = sprintf('%-8s %13s', stateNames{s}, fmt2(pr));
        row = {T, stateNames{s}, pr(1), pr(2)};
        v = NaN(numel(modes), 2);
        for m = 1:numel(modes)
            r = Tab(Tab.Tamb_C == T & strcmp(Tab.Scheduler, modes{m}) & strcmp(Tab.State, stateNames{s}), :);
            if ~isempty(r), v(m, :) = [r.RMSE(1) r.MAE(1)]; end
            txt = [txt sprintf(' %15s', fmt2(v(m, :)))]; %#ok<AGROW>
            row = [row, {v(m, 1), v(m, 2)}]; %#ok<AGROW>
        end
        iI = find(strcmp(modes, 'interp'), 1);
        for m = 1:numel(nn)
            iN = find(strcmp(modes, nn{m}), 1);
            dlt = NaN; if ~isempty(iI), dlt = 100 * (v(iN, 1) / v(iI, 1) - 1); end
            txt = [txt sprintf(' %16.1f%%', dlt)]; %#ok<AGROW>
        end
        out(fid, '%s\n', txt);
        wide(end+1, :) = row; %#ok<AGROW>
    end
end

vn = {'Tamb_C', 'State', 'Paper_RMSE', 'Paper_MAE'};
for m = 1:numel(modes)
    f = matlab.lang.makeValidName(modes{m});
    vn = [vn, {[f '_RMSE'], [f '_MAE']}]; %#ok<AGROW>
end
writetable(cell2table(wide, 'VariableNames', vn), fullfile(folder, 'table1_paper_style.csv'));
fprintf('\nWrote table1_paper_style.txt / .csv in %s\n', folder);
end

function out(fid, varargin)
% print to the Command Window and to the text file
fprintf(varargin{:});
fprintf(fid, varargin{:});
end

function s = fmt2(v)
if any(isnan(v)), s = '-'; else, s = sprintf('%.2f/%.2f', v(1), v(2)); end
end
