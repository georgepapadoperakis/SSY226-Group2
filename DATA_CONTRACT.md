# Complete training data contract

Detta beskriver den kompletterade export som behövs av 04_train_mlp.py.
Det är inte ett påstående att nuvarande temperaturfiler redan innehåller dessa data.

## State vector and current mapping

| Index (MATLAB) | Name | Unit | Current export |
|---|---|---|---|
| 1 | motT | K | T_motor_C + 273.15 |
| 2 | invT | K | T_inverter_C + 273.15 |
| 3 | dcdcT | K | T_dcdc_C + 273.15 |
| 4 | batSoc | fraction 0..1 | Missing |
| 5 | batT | K | Candidate: four-pack mean; verify COM measurement mapping |
| 6 | compPin | Pa | Missing |
| 7 | compPout | Pa | Missing |
| 8 | cabIntT | K | Missing; thermal interior mass, not cabin air |
| 9 | cabAirT | K | T_cabin_C + 273.15 |

MLP-featureordning: [Tamb_C, motT_K, invT_K, dcdcT_K, batSoc, batT_K,
compPin_Pa, compPout_Pa, cabIntT_K, cabAirT_K, Ib_A].
Tid och körd sträcka ingår inte. SOC/tryck/interiörtemperatur får inte ersättas
med nollor eller andra tillstånd för att få rätt antal kolumner.

## Control vector

| Order | Name | Unit |
|---|---|---|
| 1 | compNrpm | rpm |
| 2 | cabAirMfIn | kg/s |
| 3 | motPumpNrpm | rpm |
| 4 | batPumpNrpm | rpm |
| 5 | heaterPwr | W (model input) |
| 6 | radFanrpm | rpm |

Utöver dessa har F 56 estimatedParams och 7 controlOut-parametrar. Deras exakt
ordnade namn finns i com_metadata.json. Gamma (p1-p11) är de sista 11 elementen
i F:s parametervektor. p_fixed_horizon innehåller bara de första 63 för varje
framtida steg, konstruerade vid respektive startdriftpunkt.

## Files and arrays

Placering: Data/NN_training/complete/mlp_run_T5.npz, mlp_run_T10.npz och senare
mlp_run_T7.npz. N är antal tidsprov, normalt 3 601 för 0..3 600 s vid Ts=1 s.

| Key | Shape/content |
|---|---|
| t_s | (N,), seconds, starts at 0, constant Ts=1 |
| x | (N,9), state units as above |
| u | (N-1,6), input for transition k to k+1 |
| p_fixed_horizon | (N-1,30,63), ordered stage parameters constructed at each start k |
| Ib_A | (N,), signed according to COM's verified current convention |
| T_amb_C | scalar, -5 / -10 / -7 |
| gamma_interp | (N,11), existing interpolation evaluated at each operating point |
| state_names | exact nine names in mlp_parameter_estimator.STATE_NAMES |
| control_names | exact order from com_metadata.json |
| fixed_parameter_names | exact order from com_metadata.json |
| gamma_names | p1,...,p11 |
| reviewed_mapping | scalar True, only after verified signal/unit mapping |

Använd bara numeriska arrayer och Unicode-textarrayer i NPZ (inga Pythonobjekt).
Alla värden som används av träningsmotorn ska vara ändliga.

Parameterprognosen p_fixed_horizon[k] ska återskapa samma prognoslogik som den
granskade COM/NMPC-integrationen: termodynamiska egenskaper och överordnade
kommandon vid startpunkten, tillåtna prognoser för last/störningar och dokumenterad
fläktdynamik. Beräkna inte framtida vätskeegenskaper från referensens framtida
tillstånd; det skulle ge målläckage. De sista starterna där 30 steg inte ryms
används inte vid 30-stegsutvärdering; eventuella kortare ändfönster behöver en
uttryckligen dokumenterad slutprognos. Fyll inte saknade signaler med godtyckliga tal.

## Gamma roles in the supplied COM

| Parameter | Direct occurrence in dynamics_mexfreeNew.m |
|---|---|
| p1 | Cabin interior temperature derivative |
| p2 | Cabin air temperature derivative |
| p3 | Battery temperature derivative |
| p4 | Inverter temperature derivative |
| p5 | Motor temperature derivative |
| p6 | Refrigerant enthalpy-flow term in high-pressure dynamics |
| p7 | Battery coolant NTU term |
| p8 | Interior-to-air heat term |
| p9 | Heat term in low-pressure dynamics |
| p10 | High-pressure derivative multiplier |
| p11 | Low-pressure derivative multiplier |

Flera parametrar har indirekt inverkan genom de kopplade ekvationerna. Tabellen
är inte en identifiabilitetsanalys. DCDC:s direkta temperaturderivata har en
fast faktor 0.05 i den uppladdade modellen, inte en separat p12.

## Required validation before an experimental claim

1. Jämför exporterad CasADi-F och RK4-steg med MATLAB vid samma x,u,p.
2. Jämför flera fönster med interpolationsbaseline före MLP-träning.
3. Kontrollera gradienterna för den verkliga COM, utöver den syntetiska kontrollen.
4. Fastställ gammagränser och nominella värden med projektets kalibrering;
   observerade min/max i en tabell är inte automatiskt fysiska gränser.
5. Skala förlustens tillstånd separat: K, SOC och Pa kan inte jämföras som råa tal.
6. Håll gamma konstant över varje rollout och håll testkörningen undan från modellval.
