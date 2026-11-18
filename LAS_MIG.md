# SSY226: fran Simscape till Python

Detta ar ett forsta dataflode for projektet, anpassat till den uppladdade
RunThermalComparison.m. Det ar ingen NN-traning annu.

## Placera filerna

1. Kopiera export_logs_to_python.m till huvudmappen SSY226-Group2,
   bredvid RunThermalComparison.m.
2. Kopiera pythoncode/01_las_och_rita_data.py till projektets befintliga
   mapp pythoncode. Behall den tidigare neuralnetwork.py.

Originalfilen RunThermalComparison.m ingar inte och andras inte.

## Kor exporten i MATLAB

Oppna export_logs_to_python.m i MATLAB och tryck Run/F5. Om MATLAB fragar,
valj Change Folder till filens mapp. Du kan aven ga till projektets huvudmapp
och skriva export_logs_to_python i Command Window.

Koden anvander korningen run_2026_11_11_10_22_21 och de tre filer som borjar
med simlog_rule_T. Ingen ny simulering startas. Filen sparas som MATLAB v7,
med bara arrayer, text och strukturer, sa att SciPy kan lasa den.

Nya filer hamnar i Data/NN_exports/run_2026_11_11_10_22_21:

- rule_T5_numeric.mat
- rule_T7_numeric.mat
- rule_T10_numeric.mat

De egna exportfilerna ersatts om exporten kors igen. Kalloggarna andras inte.

## Kor kontrollen i VS Code

Valj projektets .venv med Python: Select Interpreter. Oppna den nya Pythonfilen
och valj Run Python File in Terminal. Alternativt, fran projektets huvudmapp:

```powershell
.\.venv\Scripts\python.exe .\pythoncode\01_las_och_rita_data.py
```

Programmet skriver signalernas storlek och tidssteg, kontrollerar NaN/Inf
och visar temperaturerna for kupeluft och batteripack 1 vid alla tre temperaturer.
En PNG sparas i exportmappen. Samma paket som tidigare anvands; inget nytt
paket behover installeras for dessa tva filer.

## Vad som sparas

- Alla fyra batteripackens temperaturer separat, plus kupe, motor, inverter och DCDC.
- Sex loggade effekt-/varmeflodessignaler.
- Fem separata uppskattningar av elektrisk effekt, med originalkodens antagna
  verkningsgrader tydligt lagrade som metadata. De ar inte elektriska matningar.
- Varje signals ursprungliga tidsstamplar, onskade enhet och exakta nodsokvag.
- Utetemperatur (-5/-7/-10), kallfil och regulatornamn.

Saknade signaler eller ogiltiga numeriska varden stoppar exporten med fel.
Originalkodens NaN-reservvarden anvands inte. Inga tidsvektorer jamnas ut,
sorteras, interpoleras eller slas ihop i detta steg.

## Nasta steg infor NN + COM

Temperaturer och effekter ar inte en komplett traningsuppsattning for er
parameterestimator. Vi behover matcha samtliga COM-tillstand, styrsignaler
och storningar mot loggade variabler, inklusive batteristrom for det planerade
gamma(Tamb, x, Ib). Sparad simlog kan innehalla fler signaler; den har exporten
visar inte att dessa saknas. Fordonshastighet och kompressorvarvtal ar ocksa
kandidater. Regulatorns kommandon och fysisk effekt ar olika storheter.

Val av gemensam sampling gors efter att COM:s tidsdiskretisering ar kand.
Korningar delas i traning/validering/test innan overlappande fonster skapas.

## Kontroller som har gjorts har

Pythonfilens syntax och lasning av samma strukturschema har kontrollerats med
syntetiska MAT-filer, inklusive olika tidsvektorer och en ogiltig signal.
MATLAB-koden har inte kunnat koras har: MATLAB och era verkliga loggfiler
finns inte i denna arbetsmiljo. Kor MATLAB-steget pa er dator och skicka
eventuella felmeddelanden. Alla kommentarer i kodfilerna forklarar stegen.
