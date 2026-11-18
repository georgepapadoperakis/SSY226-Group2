"""Kontrollera de 55 exporterade signalerna utan att ändra rådata.

Kör från VS Code med projektets .venv. Programmet tränar inget nätverk.
Det sparar data på 1-sekundsgrid, rapporter och jämförelsefigurer.
Tillstånd 8 lämnas NaN: kupéns samlade inredningstemperatur saknar en
verifierad definition i de exporterade mätningarna. Detta döljs inte med nollor.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
STATE_NAMES = ['motT_K', 'invT_K', 'dcdcT_K', 'batSoc', 'batT_K',
               'compPin_Pa', 'compPout_Pa', 'cabIntT_K', 'cabAirT_K']
CONTROL_NAMES = ['compNrpm', 'cabAirMfIn', 'motPumpNrpm', 'batPumpNrpm',
                 'heaterPwr', 'radFanrpm']
# Tryck från CoolProp 8.0.0, PropsSI('pcrit','R134a'). Ingen gräns för modellen
# i allmänhet; detta gäller dess NUVARANDE anrop till mättnadsegenskaper Q=0/1.
R134A_PCRIT_PA = 4059276.3737910665


def resample(signal, grid):
    """Linjär interpolation för kontrollfigurer; ingen extrapolation.

    Solverloggen kan innehålla nästan samtidiga prov kring händelser.
    Behåll det sista provet i varje grupp med högst 1 ns mellan proven.
    Denna diagnostiska grid är inte automatiskt ett färdigt val av
    interpolationsmetod för styrkommandon vid systemidentifieringen.
    """
    t = np.asarray(signal['t_s'], float).reshape(-1)
    y = np.asarray(signal['y'], float).reshape(-1)
    if t.size != y.size or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError('Tid/värde måste vara lika långa och ändliga')
    if np.any(np.diff(t) < 0):
        raise ValueError('Tiden är inte sorterad')
    keep = np.r_[np.diff(t) > 1e-9, True]
    t, y = t[keep], y[keep]
    if len(t) < 2 or t[0] > grid[0] or t[-1] < grid[-1]:
        raise ValueError('Gemensam tidsgrid täcks inte av signalen')
    return np.interp(grid, t, y)


def intervals(mask, t):
    """Första och sista gridpunkt i varje sammanhängande markerat intervall."""
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    return [[float(t[g[0]]), float(t[g[-1]])]
            for g in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)]


def read_run(file, outdir):
    context = loadmat(file, simplify_cells=True)['context']
    signals = context['signals']
    if len(signals) != 55:
        raise ValueError(f'{file.name}: väntade 55 signaler, fick {len(signals)}')
    # Gemensamt intervall: använd signalernas verkliga tidsgränser.
    start = max(float(np.asarray(s['t_s']).flat[0]) for s in signals.values())
    stop = min(float(np.asarray(s['t_s']).flat[-1]) for s in signals.values())
    t = np.arange(np.ceil(start), np.floor(stop)+1., 1.)
    aligned = {name: resample(s, t) for name, s in signals.items()}
    tamb = float(context['T_amb_C'])
    columns = ['T_motor_K', 'T_inverter_K', 'T_dcdc_K', 'SOC_pack1']
    x = np.full((len(t), 9), np.nan)
    x[:, :4] = np.column_stack([aligned[n] for n in columns])
    # Den uppladdade SLX-filens Controls/T_battery använder MAX över fyra pack.
    # Medeltemperatur kan vara användbar i en graf, men är en annan definition.
    packs = np.column_stack([aligned[f'T_bat_pack{i}_K'] for i in range(1, 5)])
    x[:, 4] = packs.max(axis=1)
    x[:, 5] = aligned['p_comp_A_Pa']
    x[:, 6] = aligned['p_comp_B_Pa']
    x[:, 8] = aligned['T_cabin_air_K']
    # T_cabInt i uppladdad SLX är kopplad till kupéluftens termiska nod.
    # Vi sparar detta alias separat; det räknas inte som en ny mätning av
    # COM:s lumpade inredningsmassa, och används inte som sann target för x8.
    alias = x[:, 8].copy()
    u = np.column_stack([aligned[n] for n in [
        'compressor_actual_rpm', 'blower_mdot_A_kg_s',
        'motor_pump_actual_rpm', 'battery_pump_actual_rpm',
        'heater_source_command_W', 'fan_actual_rpm']])
    critical = x[:, 6] >= R134A_PCRIT_PA
    # Även +/-100 Pa används i estimateParameters.m:s differenskvoter.
    sat_invalid = (x[:, 6]+100 >= R134A_PCRIT_PA) | (x[:, 5]+100 >= R134A_PCRIT_PA)
    current_residual = aligned['I_demand_physical_A'] + aligned['I_cell_pack1_A']
    command_residual = aligned['I_demand_command_A'] - aligned['I_demand_physical_A']
    # Samma temperatur/SOC i parallella pack är en kontroll, ingen förutsättning.
    socs = np.column_stack([aligned[f'SOC_pack{i}'] for i in range(1, 5)])
    native_p = signals['p_comp_B_Pa']
    native_y, native_t = np.asarray(native_p['y']), np.asarray(native_p['t_s'])
    report = {
        'file': file.name, 'ambient_C': tamb, 'source_run': str(context['source_run']),
        'signals': len(signals), 'grid_samples': len(t), 'sample_time_s': 1.,
        'time_limits_s': [float(t[0]), float(t[-1])],
        'native_samples': len(signals['T_motor_K']['y']),
        'observed_state_columns': [0, 1, 2, 3, 4, 5, 6, 8],
        'unresolved_state': 'cabIntT_K', 'training_ready': False,
        'state_mapping_model': 'uploaded SLX; historical run model identity not verified',
        'pressure_A_bar_range': [float(x[:, 5].min()/1e5), float(x[:, 5].max()/1e5)],
        'pressure_B_bar_range_native': [float(native_y.min()/1e5), float(native_y.max()/1e5)],
        'native_supercritical_samples': int((native_y >= R134A_PCRIT_PA).sum()),
        'native_supercritical_sample_intervals_s': intervals(native_y >= R134A_PCRIT_PA, native_t),
        'supercritical_grid_points': int(critical.sum()),
        'supercritical_grid_intervals_s': intervals(critical, t),
        'saturation_derivative_invalid_grid_intervals_s': intervals(sat_invalid, t),
        'current_demand_plus_cell_max_abs_A': float(np.abs(current_residual).max()),
        'command_minus_physical_current_max_abs_A': float(np.abs(command_residual).max()),
        'SOC_start_end': [float(socs[0,0]), float(socs[-1,0])],
        'max_pack_SOC_spread': float(np.ptp(socs, axis=1).max()),
        'max_pack_temperature_spread_K': float(np.ptp(packs, axis=1).max()),
        'blower_sensor_max_abs_kg_s': float(np.abs(aligned['blower_sensor_flow_kg_s']).max()),
        'blower_mdot_A_range_kg_s': [float(u[:,1].min()), float(u[:,1].max())],
        'interpolation': 'linear diagnostic interpolation; no extrapolation',
    }
    names = list(aligned)
    np.savez_compressed(outdir / f'rule_T{int(abs(tamb))}_checked.npz',
        t_s=t, x_observed=x, x8_air_alias=alias, u_diagnostic=u,
        Ib_A=aligned['I_demand_physical_A'], T_amb_C=tamb,
        signal_names=np.array(names), signal_values=np.column_stack([aligned[n] for n in names]),
        state_names=np.array(STATE_NAMES), control_names=np.array(CONTROL_NAMES),
        observable_state_mask=np.array([1,1,1,1,1,1,1,0,1], dtype=bool),
        saturation_domain_valid=~sat_invalid,
        training_ready=False, reviewed_mapping=False)
    print(f"{file.name}: {len(signals)} signals, {len(t)} grid points, "
          f"8/9 separately defined COM states; supercritical grid points={critical.sum()}")
    return t, aligned, report


def plot_runs(runs, output):
    colors = {-5.: 'tab:blue', -7.: 'tab:orange', -10.: 'tab:green'}
    # Alla exporterade temperaturer visas, två diagram per bild som tidigare.
    panels = [
        ('T_cabin_air_K', 'Cabin air temperature', 'Temperature [°C]'),
        ('T_bat_pack1_K', 'Battery pack 1 temperature', 'Temperature [°C]'),
        ('T_bat_pack2_K', 'Battery pack 2 temperature', 'Temperature [°C]'),
        ('T_bat_pack3_K', 'Battery pack 3 temperature', 'Temperature [°C]'),
        ('T_bat_pack4_K', 'Battery pack 4 temperature', 'Temperature [°C]'),
        ('T_motor_K', 'Motor temperature', 'Temperature [°C]'),
        ('T_inverter_K', 'Inverter temperature', 'Temperature [°C]'),
        ('T_dcdc_K', 'DC/DC converter temperature', 'Temperature [°C]'),
        ('T_glass_K', 'Glass temperature', 'Temperature [°C]'),
        ('T_doors_K', 'Doors temperature', 'Temperature [°C]'),
        ('T_roof_K', 'Roof temperature', 'Temperature [°C]'),
        ('SOC_pack1', 'Battery pack 1 state of charge', 'SOC [%]'),
        ('p_comp_A_Pa', 'Compressor inlet pressure — port A', 'Pressure [bar]'),
        ('p_comp_B_Pa', 'Compressor outlet pressure — port B', 'Pressure [bar]'),
    ]
    for index in range(0, len(panels), 2):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout='constrained')
        for ax, (name, title, unit) in zip(axes, panels[index:index+2]):
            for t, signals, r in runs:
                y = signals[name]
                if name.endswith('_K'): y = y-273.15
                elif name.endswith('_Pa'): y = y/1e5
                elif name.startswith('SOC'): y = y*100
                ax.plot(t, y, color=colors[r['ambient_C']], label=f"Ambient {r['ambient_C']:g} °C")
            if name == 'p_comp_B_Pa':
                ax.axhline(R134A_PCRIT_PA/1e5, color='black', linestyle='--',
                           label='R134a critical pressure (CoolProp)')
            ax.set(title=title, xlabel='Time [s]', ylabel=unit)
            ax.grid(alpha=.4); ax.legend()
        fig.savefig(output / f'context_comparison_{index//2+1:02d}.png', dpi=150)
        plt.close(fig)
    # Visa skillnaden mellan kupéluften och de tre väggtemperaturerna.
    t, signals, r = next(run for run in runs if run[2]['ambient_C'] == -5.)
    fig, ax = plt.subplots(figsize=(12,5), layout='constrained')
    for n, label in [('T_cabin_air_K','Cabin air / wired T_cabInt alias'),
                     ('T_glass_K','Glass'),('T_doors_K','Doors'),('T_roof_K','Roof')]:
        ax.plot(t, signals[n]-273.15, label=label)
    ax.set(title='Cabin state definition check — ambient −5 °C', xlabel='Time [s]', ylabel='Temperature [°C]')
    ax.grid(alpha=.4); ax.legend()
    fig.savefig(output / 'cabin_state_definition.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path,
                        help='Mappen med de tre context_rule_T*.mat; annars söks den senaste gemensamma körningen')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'Data'/'NN_training'/'context_checked')
    parser.add_argument('--report-dir', type=Path,
                        default=ROOT/'reports'/'context_validation')
    args = parser.parse_args()
    if args.input_dir:
        source = args.input_dir
    else:
        raw = ROOT/'Data'/'NN_training'/'context_raw'
        groups = [p.parent for p in raw.rglob('context_rule_T5.mat')
                  if all((p.parent/f'context_rule_T{i}.mat').is_file() for i in (5,7,10))]
        if not groups:
            raise FileNotFoundError(f'Ingen komplett grupp med tre context-filer under {raw}')
        source = max(set(groups), key=lambda p: (p.name, str(p)))
    files = [source/f'context_rule_T{i}.mat' for i in (5,7,10)]
    if not all(p.is_file() for p in files):
        raise FileNotFoundError(f'Alla tre context-filer måste finnas i {source}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    runs = [read_run(p, args.output_dir) for p in files]
    plot_runs(runs, args.report_dir)
    report = {'training_ready': False,
              'reason': 'cabIntT_K is not a separately verified reference; saturation-property domain exceptions exist',
              'r134a_critical_pressure_Pa': R134A_PCRIT_PA,
              'runs': [run[2] for run in runs]}
    path = args.report_dir/'context_readiness.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved checked numeric data:', args.output_dir)
    print('Saved figures and report:', args.report_dir)
    print('No training performed. cabIntT_K remains unresolved; raw files unchanged.')


if __name__ == '__main__':
    main()
