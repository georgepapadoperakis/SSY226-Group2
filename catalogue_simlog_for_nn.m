function catalogue_simlog_for_nn()
% Lista riktiga signalvägar utan att gissa namn på SOC, ström, tryck osv.
% Placera filen i huvudmappen och kör catalogue_simlog_for_nn i MATLAB.
% Vi behöver bara en körning för att kartlägga trädstrukturen.
projektmapp = fileparts(mfilename('fullpath'));
fil = fullfile(projektmapp, 'Results', 'run_2026_11_11_10_22_21', 'simlog_rule_T5.mat');
S = load(fil, 'simlog');
rows = walk_node(S.simlog, '');
assert(~isempty(rows), 'Inga loggade serier hittades');
tabell = cell2table(rows, 'VariableNames', {'NodePath','Unit','Samples','Rows','Columns'});
utdata = fullfile(projektmapp, 'Data', 'NN_training');
if ~isfolder(utdata), mkdir(utdata); end
utfil = fullfile(utdata, 'signal_catalogue.csv');
writetable(tabell, utfil);
fprintf('Hittade %d serier. Katalog sparad:\n%s\n', height(tabell), utfil);
end


function rows = walk_node(nod, sokvag)
% Slutnoder har en series med antal prov och enhet. Värden behöver inte läsas.
rows = cell(0, 5);
serie = nod.series;
if serie.points > 0
    dim = serie.dimension;
    rows = {sokvag, char(serie.unit), double(serie.points), double(dim(1)), double(dim(2))};
end
% properties visar både metadata och dynamiska barnnoder.
namn = properties(nod);
for k = 1:numel(namn)
    if ismember(namn{k}, {'id','savable','exportable','series'}), continue; end
    barn = nod.(namn{k});
    if isa(barn, 'simscape.logging.Node')
        if isempty(sokvag), ny = namn{k}; else, ny = [sokvag '.' namn{k}]; end
        rows = [rows; walk_node(barn, ny)]; %#ok<AGROW>
    end
end
end
