"""Steg 2: las MATLAB-exporten och kontrollera signalerna fore NN-traning.

Placera filen i SSY226-Group2/pythoncode och kor den i projektets .venv.
Alla kommentarer forklarar det som gors. Ingen omsampling eller traning sker.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


# __file__ ar denna fils sokvag. parents[1] gar upp tva mappnivaer:
# fil -> pythoncode -> SSY226-Group2.
PROJEKTMAPP = Path(__file__).resolve().parents[1]
KORNING = "run_2026_11_11_10_22_21"
EXPORTMAPP = PROJEKTMAPP / "Data" / "NN_exports" / KORNING


def las_och_kontrollera(fil):
    """Las en korning och kontrollera alla exporterade signaler."""
    # simplify_cells=True gor MATLAB-strukturer till Python-dictionaries.
    # En dictionary innehaller namngivna poster, som hamtas med ["namn"].
    data = loadmat(fil, simplify_cells=True)["run_data"]
    assert int(data["schema_version"]) == 1, f"Okant exportformat: {fil}"
    signaler = data["signals"]
    print(f"\n{fil.name}: T_amb = {float(data['T_amb_C']):g} degC")

    for namn, signal in signaler.items():
        # MATLAB har rad-/kolumnvektorer. I Python anvander vi har 1D-arrayer.
        t = np.asarray(signal["t_s"], dtype=float).reshape(-1)
        y = np.asarray(signal["y"], dtype=float).reshape(-1)
        if t.size < 2 or t.size != y.size:
            raise ValueError(f"Fel storlek i {fil.name}, {namn}")
        if not (np.isfinite(t).all() and np.isfinite(y).all()):
            raise ValueError(f"NaN eller Inf i {fil.name}, {namn}")
        if (np.diff(t) < 0).any() or t[-1] <= t[0]:
            raise ValueError(f"Ogiltig tidsvektor i {fil.name}, {namn}")

        signal["t_s"] = t
        signal["y"] = y
        dt = np.diff(t)
        # Medianen sammanfattar tidsstegen; den innebar inte jamn sampling.
        print(
            f"  {namn:28s} n={t.size:7d}, "
            f"tid={t[0]:.3f}..{t[-1]:.3f} s, "
            f"dt min/median/max={dt.min():.5g}/{np.median(dt):.5g}/{dt.max():.5g} s"
        )

    return data


def rita_alla_signaler(korningar):
    """Visa alla 19 exporterade signaler, med högst två diagram per fönster."""

    # Varje rad innehåller signalens namn och diagrammets engelska rubrik.
    # Signalnamnen till vänster måste behållas exakt som i exportfilen.
    # Uppskattade elektriska effekter märks med "estimated".
    paneler = [
        ("T_cabin_C", "Cabin air temperature"),
        ("T_bat_pack1_C", "Battery pack 1 temperature"),
        ("T_bat_pack2_C", "Battery pack 2 temperature"),
        ("T_bat_pack3_C", "Battery pack 3 temperature"),
        ("T_bat_pack4_C", "Battery pack 4 temperature"),
        ("T_motor_C", "Motor temperature"),
        ("T_inverter_C", "Inverter temperature"),
        ("T_dcdc_C", "DC/DC converter temperature"),

        ("P_comp_mech_W", "Compressor - mechanical power"),
        ("P_motor_pump_mech_W", "Motor pump - mechanical power"),
        ("P_battery_pump_mech_W", "Battery pump - mechanical power"),
        ("P_blower_logged_W", "Cabin blower - logged power"),
        ("P_fan_mech_W", "Radiator fan - mechanical power"),
        ("Q_heater_W", "Heater - heat flow rate"),

        ("P_comp_elec_est_W",
         "Compressor - estimated electrical power"),
        ("P_motor_pump_elec_est_W",
         "Motor pump - estimated electrical power"),
        ("P_battery_pump_elec_est_W",
         "Battery pump - estimated electrical power"),
        ("P_blower_elec_est_W",
         "Cabin blower - estimated electrical power"),
        ("P_fan_elec_est_W",
         "Radiator fan - estimated electrical power"),
    ]

    # Samma utetemperatur får samma färg i samtliga diagram.
    farger = {
        -5: "tab:blue",
        -7: "tab:orange",
        -10: "tab:green",
    }

    # Mappen där diagrammen sparas.
    bildmapp = EXPORTMAPP / "bilder"

    # Kontrollera forst att samtliga forvantade signaler finns och har ratt enhet.
    # Da avbryts korningen tydligt om exporten ar ofullstandig.
    for data in korningar:
        for namn, _ in paneler:
            if namn not in data["signals"]:
                raise ValueError(
                    f"Saknar {namn} vid {data['T_amb_C']} °C. "
                    "Kontrollera att MATLAB-exporten ar fullstandig."
                )
            vantad_enhet = "degC" if namn.startswith("T_") else "W"
            if data["signals"][namn]["unit"] != vantad_enhet:
                raise ValueError(f"Ovantad enhet for {namn}: vantade {vantad_enhet}")

    bildmapp.mkdir(parents=True, exist_ok=True)

    # range(0, ..., 2) tar tva signaler i taget: 0-1, 2-3, 4-5 och sa vidare.
    # Sista gruppen innehaller bara en signal eftersom antalet ar udda.
    for start in range(0, len(paneler), 2):
        grupp = paneler[start:start + 2]
        nummer = start // 2 + 1

        # subplots skapar figurfonstret och dess diagramytor.
        # squeeze=False gor att axlar alltid ar en 2D-array, aven med ett diagram.
        # constrained_layout ger plats for rubriker, etiketter och forklaringar.
        fig, axlar = plt.subplots(
            len(grupp), 1, figsize=(13, 8 if len(grupp) == 2 else 4.5),
            squeeze=False, constrained_layout=True,
        )
        fig.canvas.manager.set_window_title(f"SSY226 – diagram {nummer:02d}")

        for axel, (namn, rubrik) in zip(axlar.flat, grupp):
            for data in korningar:
                signal = data["signals"][namn]
                temperatur = float(data["T_amb_C"])

                # Anvand varje signals egen tidsvektor. Ingen omsampling sker.
                axel.plot(
                    signal["t_s"], signal["y"],
                    color=farger[temperatur], linewidth=1.5,
                    label=f"Utetemperatur {temperatur:g} °C",
                )

            axel.set_title(rubrik)
            axel.set_xlabel("Tid [s]")
            if namn.startswith("T_"):
                axel.set_ylabel("Temperatur [°C]")
            elif namn == "Q_heater_W":
                axel.set_ylabel("Värmeflöde [W]")
            else:
                axel.set_ylabel("Effekt [W]")
            axel.grid(True)
            axel.legend(loc="best")

        # Spara en PNG av samma figur som visas i fonstret.
        bildfil = bildmapp / f"{nummer:02d}_{grupp[0][0]}.png"
        fig.savefig(bildfil, dpi=180)
        print(f"Sparad bild: {bildfil}")

    print(f"\n{len(paneler)} signaler visas i {(len(paneler) + 1) // 2} figurfönster.")
    print(f"Alla diagram finns även som PNG i: {bildmapp}")
    # show ligger EFTER loopen: alla fonster oppnas vid samma korning.
    # Om show lag inne i loopen kunde du behova stanga ett fonster i taget.
    plt.show()


def main():
    # Anvand bestamda filnamn sa att exportfiler fran andra korningar inte blandas in.
    korningar = []
    for temperatur in (-5, -7, -10):
        fil = EXPORTMAPP / f"rule_T{abs(temperatur)}_numeric.mat"
        if not fil.is_file():
            raise FileNotFoundError(
                f"Saknar {fil}\nKor export_logs_to_python i MATLAB forst."
            )
        data = las_och_kontrollera(fil)
        if float(data["T_amb_C"]) != temperatur:
            raise ValueError(f"Temperaturmetadata stammer inte med filnamnet: {fil}")
        korningar.append(data)

    print("Data kunde lasas och kontrolleras. Inget natverk har tranats annu.")
    rita_alla_signaler(korningar)


# main() kors bara nar vi startar just den har filen.
# Det gor att funktionerna senare kan importeras fran andra Pythonfiler.
if __name__ == "__main__":
    main()
