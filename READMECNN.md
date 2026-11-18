# SSY226: MLP parameter scheduling preparation

Detta paket förbereder det upplägg ni valt: ett feedforward-MLP, tränat med
backpropagation genom COM. Kommentarerna är på svenska; signalnamn och
rapporter använder engelska. Era originalfiler ändras inte.

## What is ready

- MLP: 11 -> 32 -> 32 -> 11, Tanh i två dolda lager, totalt 1 803 träningsbara tal.
- Indata: utetemperatur, COM:s nio tillstånd, batteriström. Ingen tid eller sträcka.
- Utdata: p1-p11 i samma ordning som i create_dynamics_nmpcNew.m/prepParaVec.m.
- Begränsad utdata med granskade parametervärden som måste fyllas i före träning.
- Normalisering beräknad på enbart träningsdelens driftpunkter.
- CasADi-brygga som bevarar gradienter genom RK4 och flerstegssimulering.
- Träningsmotor där gamma skattas en gång per fönster och hålls över horisonten.
- Kontroll av de tre verkliga MAT-filerna och förberedelse av fem kända tillstånd.
- MATLAB-verktyg för att katalogisera kvarvarande signaler och exportera COM.

## Important distinction

MLP är arkitekturen. Backpropagation beräknar hur förlusten påverkas av vikterna.
AdamW använder dessa gradienter för att uppdatera vikterna. Förlusten i den riktiga
träningsmotorn är normaliserat tillståndsfel över flera steg, inte passning av
p1-p11 som huvudsakliga etiketter. En liten, konfigurerbar prior mot interpolerade
parametrar hjälper när data inte entydigt bestämmer gamma.

Modellstorlek, Tanh, AdamW och träningsinställningar är startval, inte jämförda
eller optimerade lösningar för era verkliga data. Parametrarnas fysiska gränser
är inte gissade. De står därför som null i config tills de har fastställts.

## Install and place files

1. Packa upp paketet.
2. Kopiera innehållet till befintliga SSY226-Group2. Slå ihop mappen pythoncode
   med den ni redan har. De nya filnamnen ersätter inte neuralnetwork.py.
3. export_com_for_python.m, catalogue_simlog_for_nn.m och mlp_training_config.json
   ska ligga i huvudmappen. Pythonfilerna ska ligga i pythoncode.
4. Använd er befintliga .venv. Installera Python-CasADi i den miljön:

```powershell
.\.venv\Scripts\python.exe -m pip install casadi==3.7.1
```

Versionen matchar namnet på MATLAB-CasADi i ert projekt. export_com_for_python
skriver ut faktisk CasADi-version i metadata; om den avviker, matcha Python mot
den versionen. MATLAB:s CoolProp-koppling behöver inte ändras för dessa steg.

## Step 1: prepare the available real data

```powershell
.\.venv\Scripts\python.exe .\pythoncode\02_forbered_mlp_data.py
```

Förväntat: 3 601 tidpunkter, 1 s sampling, 5/9 tillstånd tillgängliga per körning.
Filerna sparas i Data/NN_training/prepared_partial och heter uttryckligen PARTIAL.
De saknade tillstånden lämnas som NaN med separat mask; inga nollor fylls i.
Den riktiga träningsmotorn accepterar inte detta ofullständiga schema.

Resamplingen i detta steg gäller bara temperaturer (linjär interpolation).
Utetemperaturen är känd från metadata. Batteritemperaturens första kandidat är
medelvärdet av fyra tidsjusterade pack, i linje med er plottningskod; bekräfta
att det är rätt referens för COM:s x5 innan träning. Inga derivator uppskattas
från de nästan upprepade lösartiderna.

## Step 2: verify MLP and backpropagation

```powershell
.\.venv\Scripts\python.exe .\pythoncode\03_kontrollera_backprop.py
```

Det här är ett separat tekniskt test på en SYNTETISK dynamik med samma dimensioner.
Det skriver PASS när den numeriska gradientkontrollen fungerar och vikternas
uppdatering minskar simuleringsfelet. Det tränar eller validerar inte bilmodellen.
Testets parametervärden och gränser är enbart testvärden.

## Step 3: catalogue the saved vehicle log in MATLAB

Kör från projektets huvudmapp:

```matlab
catalogue_simlog_for_nn
export_com_for_python
```

Ingen av funktionerna startar en ny Simscape-simulering. Första funktionen skriver
Data/NN_training/signal_catalogue.csv med riktiga signalvägar, enheter och antal
prov. Den andra skapar Data/NN_training/COM/vehicle_com.casadi och com_metadata.json.
MATLAB stegen måste köras på er dator eftersom loggarna och MATLAB finns där.

Skicka signal_catalogue.csv, com_metadata.json, vehicle_com.casadi och
tunedVariables_minus5_minus10.mat från projektets Data-mapp för nästa steg.
Den sistnämnda filen laddas redan av init_nmpc.m. Den behövs för att förstå
parametrarnas skala och bygga korrekt interpolationsbaseline, inte för att
okritiskt använda interpolerade värden som slutliga etiketter.

## What is still missing before vehicle training

Fyra av nio COM-tillstånd saknas i den nuvarande exporten: SOC, kompressorns
inloppstryck och utloppstryck samt kupéns inredningstemperatur. Batteriström,
de sex styrsignalerna och flera termodynamiska/överordnade parametrar saknas också.
Signalerna kan finnas i den sparade Simscape-loggen; nuvarande export avgör inte det.

De exporterade mekaniska effekterna är inte samma storheter som kompressorvarvtal,
pumpvarvtal eller fläktvarvtal. Värmarens loggade värmeflöde är inte automatiskt
COM:s heaterPwr. Gissa inte dessa kopplingar eller fyll saknade signaler med nollor.

COM tar även 63 givna est/ctrl-parametrar. Ordningen exporteras i metadata.
CoolProp-egenskaper/överordnad logik måste konstrueras eller loggas konsekvent
med den granskade COM-körningen. Bryggan håller dessa som givna data per steg,
precis som F(x,u,p) används med Pstage i controller-koden; inga CoolProp-gradienter
har införts genom en ny, återkopplad egenskapsberäkning.

## Later: run the vehicle training

Detta görs efter att DATA_CONTRACT.md uppfyllts, modellens enstegsprediktion har
kontrollerats mot MATLAB och parametervärdena i config granskats.

```powershell
.\.venv\Scripts\python.exe .\pythoncode\04_train_mlp.py
```

Med nuvarande config stoppar kommandot avsiktligt: fysiska parametervärden saknas.
Första horisonten är 5 s för numerisk felsökning. Slutlig jämförelse ska göras
med N=30 och Ts=1 s från init_nmpc.m. Kontrollera alltid att en RK4-substep
har tillräcklig noggrannhet/stabilitet i de studerade driftfallen.

Träning använder -5 och -10 C före 2 700 s. Validering använder samma två körningar
efter 2 730 s. En 30 s lucka skiljer delarna. Detta är preliminär tidsvalidering,
inte validering på oberoende körcykler. -7 C hålls undan och är ett interpolationstest.
Efter att inställningarna valts kan ett sluttest begäras med --evaluate-test.
Träningsförlust och valideringsfel sparas i JSON, tillsammans med checkpointen.

## Validation performed here

- Alla tre verkliga MAT-filer lästes, samtliga 19 signaler var ändliga.
- Tidsvektorerna är identiska mellan exporterade signaler inom varje körning.
- 5 kända tillstånd förbereddes på 1 s-grid för alla tre körningar.
- Numerisk gradcheck verifierade gradienter för både x och gamma genom flera
  RK4-steg med syntetisk testdynamik.
- Syntetiskt flerstegsfel minskade från cirka 1.20418 till 0.00001851.
- MATLAB-export och fordonsmodellens verkliga gradienter har inte kunnat
  köras här, eftersom MATLAB och råa simlog-filer inte har bifogats.

## Source-code observations to review

Den befintliga estimateParameters.m beräknar uteluftsegenskaper vid konstant
envAirT=263.15 K. Det innebär att just de egenskaperna inte varierar med -5/-7 C
trots att envT gör det. Ändra inte detta samtidigt som MLP införs utan att
dokumentera och testa modelländringen separat; annars kan jämförelsen blandas ihop.

DCDC-temperaturens verifierade loggväg är DCDC.DC_DC_Converter.T.
Originalets stats-funktion fångade den tidigare felaktiga vägen och gav NaN.
