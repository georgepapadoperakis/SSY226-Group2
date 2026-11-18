# Första steget:
# 1. Kontrollera Python och de installerade paketen.
# 2. Hitta projektets resultatfiler.
#
# Vi läser ännu inte in mätvärden och tränar inget nätverk.


# Path hjälper oss att hantera mappar och filer.
# Det fungerar även om sökvägen innehåller mellanslag.
from pathlib import Path

# sys ger information om Python som kör programmet.
import sys

# Importera paketen. Om något saknas stannar programmet här
# och skriver exempelvis "ModuleNotFoundError".
import torch
import numpy as np
import pandas as pd
import matplotlib
import scipy


# --------------------------------------------------
# DEL 1: Kontrollera Pythonmiljön
# --------------------------------------------------

print("Python som kör programmet:")
print(sys.executable)

# Sökvägen ovan ska innehålla .venv\Scripts\python.exe.

print("\nPaketversioner:")
print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("SciPy:", scipy.__version__)


# --------------------------------------------------
# DEL 2: Hitta projektets huvudmapp
# --------------------------------------------------

# __file__ är sökvägen till den här Pythonfilen.
# resolve() gör den till en fullständig sökväg.
denna_fil = Path(__file__).resolve()

# Filen ligger i pythoncode.
kodmapp = denna_fil.parent

# Ett steg ovanför pythoncode ligger SSY226-Group2.
projektmapp = kodmapp.parent

# Skapa sökvägen till Results.
# "/" används här för att sätta ihop sökvägar.
resultatmapp = projektmapp / "Results"

print("\nProjektmapp:")
print(projektmapp)

print("\nResultatmapp:")
print(resultatmapp)


# --------------------------------------------------
# DEL 3: Hitta MAT-filer
# --------------------------------------------------

# Kontrollera att Results faktiskt finns.
if not resultatmapp.is_dir():
    raise FileNotFoundError(
        f"Hittar inte resultatmappen: {resultatmapp}"
    )

# rglob söker även i alla undermappar till Results.
# *.mat betyder alla filer vars namn slutar med .mat.
# sorted gör att filerna får en bestämd ordning.
matfiler = sorted(resultatmapp.rglob("*.mat"))

print(f"\nAntal hittade MAT-filer: {len(matfiler)}")

# Gå igenom filerna en i taget.
for fil in matfiler:
    # Visa sökvägen från Results, så att även undermappen syns.
    relativ_sokvag = fil.relative_to(resultatmapp)
    print(" -", relativ_sokvag)

if not matfiler:
    print("Inga MAT-filer hittades. Kontrollera var de är sparade.")

print("\nKontrollen är klar.")

# --------------------------------------------------
# DEL 4: Undersök MAT-filerna med h5py
# --------------------------------------------------

import h5py


# Använd listan med MAT-filer som skapades i DEL 3.
for fil in matfiler:

    print("\n" + "=" * 60)
    print("Undersöker:", fil.name)

    # Öppna filen i läsläge.
    # "r" betyder read: programmet ändrar inte filens innehåll.
    # with ser till att filen stängs efter detta kodblock.
    with h5py.File(fil, "r") as innehall:

        # En HDF5-fil kan innehålla:
        # - grupper: behållare som liknar mappar
        # - dataset: lagrade arrayer eller annan data
        #
        # items() ger oss namn och objekt på filens översta nivå.
        for namn, objekt in innehall.items():

            print("\nNamn:", namn)

            # MATLAB kan ha sparat information om variabelns klass.
            # Om uppgiften saknas använder vi texten "okänd".
            matlabklass = objekt.attrs.get("MATLAB_class", "okänd")

            # Text kan vara lagrad som bytes.
            # decode omvandlar då innehållet till vanlig Pythontext.
            if isinstance(matlabklass, bytes):
                matlabklass = matlabklass.decode(
                    "utf-8", errors="replace"
                )

            print("  MATLAB-klass:", matlabklass)

            # isinstance kontrollerar vilken sorts objekt vi har.
            if isinstance(objekt, h5py.Dataset):
                print("  HDF5-typ: dataset")
                print("  Lagrade dimensioner:", objekt.shape)
                print("  Lagrad datatyp:", objekt.dtype)

            elif isinstance(objekt, h5py.Group):
                print("  HDF5-typ: grupp")

                # Visa högst 15 namn inuti gruppen.
                # Det begränsar utskriften om filen innehåller
                # många interna MATLAB-objekt.
                undernamn = list(objekt.keys())
                print("  Antal underobjekt:", len(undernamn))
                print("  Första underobjekten:", undernamn[:15])

print("\nUndersökningen med h5py är klar.")