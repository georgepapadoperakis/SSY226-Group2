% Run NMPC vs rule-based comparison at multiple ambient temperatures:
RunThermalComparison('both', [-5, -7, -10])

% NMPC only at a single temperature:
RunThermalComparison('nmpc', -10)