"""Första COM+MLP-experiment med åtta uppmätta tillstånd och ett latent tillstånd.

Viktigt: cabIntT_K är en KAUSAL MODELLUPPSKATTNING, aldrig en målvariabel.
Den tas fram från tidigare uppmätt kupéluft och befintlig gamma-interpolation.
Det innebär ett nytt, dokumenterat experimentupplägg. Det ger inte nio
oberoende mätningar och rättar inte automatiskt controller-kopplingen i SLX.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from thermo_properties import parameters

ROOT = Path(__file__).resolve().parents[2]
DRIVING_STATE = [7,8,4,1,0,2,4,8,5,6,5]  # MATLAB-index minus ett


class GammaInterpolation:
    """Python-motsvarighet till prepParaVec: linear + nearest utanför höljet.

    Punkter med exakt samma koordinat slås ihop med medelvärde.
    Triangulering och randfall ska jämföras med MATLAB numeriskt
    före driftsättning. Funktionen innehåller ingen ny inlärning.
    """
    def __init__(self, file):
        tv = loadmat(file, simplify_cells=True)['tunedVariables']
        self.fields = []
        states = np.asarray(tv['states'], float)
        ambient = np.asarray(tv['ambient'], float).ravel()
        values = np.asarray(tv['parameters'], float)
        for i, j in enumerate(DRIVING_STATE):
            points = np.column_stack([ambient, states[:,j]])
            unique, inverse = np.unique(points, axis=0, return_inverse=True)
            z = np.bincount(inverse, weights=values[:,i])/np.bincount(inverse)
            self.fields.append((LinearNDInterpolator(unique,z), NearestNDInterpolator(unique,z)))

    def one(self, parameter, ambient, state):
        linear, nearest = self.fields[parameter]
        out = np.asarray(linear(ambient,state), float)
        # Undvik olika skalär-/vektorfall i interpolatorns returntyp.
        return np.where(np.isfinite(out), out, nearest(ambient,state))

    def __call__(self, x, ambient):
        return np.column_stack([self.one(i,ambient,x[:,j]) for i,j in enumerate(DRIVING_STATE)])


def estimate_interior(t, air, ambient, prior):
    """Enkel modellprediktor från cabMdl, inte en mätbaserad korrigerad observer.

    Vid k används T_air[k] och den gamla parameterfunktionen. Därför kan
    uppskattningen vid k inte innehålla framtida referenstemperaturer.
    Börjar vid samma temperatur som väggarnas kalla initialtillstånd.
    """
    r = (.01217*(1/(40*3.5)+.002/(.96*3.5))
         + .15076*(1/(40*4)+.03/(.08*4))
         + .63668*(1/(40*2)+.02/(.08*2)))
    thermal_mass = 36.4*840+75*500+30*500
    z = np.empty(len(t)); z[0] = ambient+273.15
    for k in range(len(t)-1):
        gamma1 = float(prior.one(0, ambient, z[k]))
        c = gamma1*42237./(2287.85*thermal_mass+1e-6)/(r+1e-6)
        # Precis samma skalära delmodell som COM. T_air och gamma hålls
        # under steget, och RK4 används precis som i COM-bryggan.
        rhs = lambda v: c*(ambient+273.15+air[k]-2*v)
        h = t[k+1]-t[k]
        a = rhs(z[k]); b = rhs(z[k]+h*a/2)
        c2 = rhs(z[k]+h*b/2); d = rhs(z[k]+h*c2)
        z[k+1] = z[k]+h*(a+2*b+2*c2+d)/6
    return z


def previous(signal, grid):
    """Håll senast loggade styrvärde. Behåll sista provet nära en händelse."""
    t, y = np.asarray(signal['t_s']), np.asarray(signal['y'])
    keep = np.r_[np.diff(t)>1e-9, True]
    return interp1d(t[keep],y[keep],kind='previous',bounds_error=True)(grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, help='Mappen med context_rule_T*.mat')
    parser.add_argument('--checked-dir', type=Path, default=ROOT/'Data'/'NN_training'/'context_checked')
    parser.add_argument('--com-dir', type=Path, default=ROOT/'Data'/'NN_training'/'COM')
    parser.add_argument('--tuning-file', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'Data'/'NN_training'/'experiment_8observed')
    args = parser.parse_args()
    if args.tuning_file is None:
        matches = sorted(ROOT.rglob('tunedVariables_minus5_minus10.mat'), key=lambda p:len(p.parts))
        if not matches: raise FileNotFoundError('Ange --tuning-file till er befintliga parametertabell')
        args.tuning_file = matches[0]
    if args.input_dir is None:
        groups = [p.parent for p in (ROOT/'Data'/'NN_training'/'context_raw').rglob('context_rule_T5.mat')
                  if all((p.parent/f'context_rule_T{i}.mat').exists() for i in (5,7,10))]
        if not groups: raise FileNotFoundError('Hittar inte gruppen med tre råa context-filer')
        args.input_dir = max(set(groups), key=lambda p:(p.name,str(p)))
    meta = json.loads((args.com_dir/'com_metadata.json').read_text(encoding='utf-8'))
    prior = GammaInterpolation(args.tuning_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'status':'PRELIMINARY_EXPERIMENT', 'independent_observed_states':8,
              'latent_state':'cabIntT_K', 'latent_target_weight':0,
              'latent_method':'causal COM cabin-interior predictor, initialized at ambient, prior gamma1',
              'thermo_python_matlab_parity_verified':False,
              'interpolant_python_matlab_parity_verified':False,
              'actuator_assumption':'sample-hold physical speed/flow/heater; logical cold-mode supervisor',
              'forecasts':'recorded external disturbances; not an online forecasting performance claim',
              'table_sha256':hashlib.sha256(args.tuning_file.read_bytes()).hexdigest(),
              'excluded_startup_before_s':30., 'runs':[]}
    for case in (5,7,10):
        print(f'Preparing T{case}...',flush=True)
        raw_file = args.input_dir/f'context_rule_T{case}.mat'
        context = loadmat(raw_file,simplify_cells=True)['context']; raw = context['signals']
        data = dict(np.load(args.checked_dir/f'rule_T{case}_checked.npz', allow_pickle=False))
        t,x,tamb = data['t_s'], data['x_observed'].copy(), float(data['T_amb_C'])
        signals = dict(zip(data['signal_names'], data['signal_values'].T))
        for name in ('T_glass_K','T_doors_K','T_roof_K'):
            if not np.isclose(signals[name][0], tamb+273.15,atol=1e-6):
                raise ValueError('Kallstart vid ambient kan inte antas för inredningens initialvärde')
        x[:,7] = estimate_interior(t,x[:,8],tamb,prior)
        gamma = prior(x,tamb)
        # Aktuatorer till COM hålls konstanta under varje 1-sekundssteg.
        # Använd de aktuella signalerna, inte den nollställda blower-sensorn.
        unames = ['compressor_actual_rpm','blower_mdot_A_kg_s','motor_pump_actual_rpm',
                  'battery_pump_actual_rpm','heater_source_command_W','fan_actual_rpm']
        u = np.column_stack([previous(raw[n],t) for n in unames])
        ib = signals['I_demand_physical_A']
        drive = context['drive_cycle']
        speed = np.interp(t, drive['t_s'], drive['values_native']) # km/h, verifierat SLX
        heat = np.column_stack([signals[n] for n in ['Q_motor_source_W','Q_inverter_source_W','Q_dcdc_source_W']])
        # Frys vätskeegenskaper vid fönstrets början. Framtida tillstånd
        # används aldrig för att uppdatera dem inne i prediktionshorisonten.
        valid = (t>=30)&data['saturation_domain_valid']
        fixed = np.full((len(t),63),np.nan)
        ids = np.flatnonzero(valid)
        # Originalsupervisorn är i värmepumpsläge efter uppstart i dessa kalla
        # körningar. Bekräfta att ventil-/blandningssignaler stöder antagandet.
        if np.any(signals['evap_valve_S_m'][ids] > 1e-8) or np.any(signals['chiller_valve_S_m'][ids]>1e-8):
            raise ValueError('Experimentets antagande om stängda kylventiler stöds inte')
        if np.min(signals['blend_AB_area_m2'][ids]) < .0359:
            raise ValueError('Experimentets antagande om full värmepumpsblandning stöds inte')
        if np.min(signals['serial_valve1_S_m'][ids]) < .00629:
            raise ValueError('Experimentets antagande om seriell koppling stöds inte')
        fields = parameters(x[ids],tamb,speed=speed[ids],ib=ib[ids],heat=heat[ids].T,
                            fan=u[ids,5],ctrl=(0.,1.,1.,0.,0.,2.,1.))
        for j,n in enumerate(meta['fixed_parameter_names']):fixed[ids,j]=fields[n]
        valid &= np.isfinite(fixed).all(axis=1)
        # 5 givna externa storheter per tidpunkt; deras referensprognos är
        # tillåten för identifiering, men är inte en implementerad prognosmodell.
        exogenous = np.column_stack([speed,ib,heat])
        mask = np.array([1,1,1,1,1,1,1,0,1],bool)
        np.savez_compressed(args.output_dir/f'experiment_T{case}.npz',t_s=t,x=x,
            u=u[:-1],fixed_at_start=fixed,exogenous=exogenous,gamma_prior=gamma,
            Ib_A=ib,T_amb_C=tamb,target_mask=mask,valid_start=valid,
            state_names=np.array(meta['state_names']),control_names=np.array(meta['control_names']),
            fixed_parameter_names=np.array(meta['fixed_parameter_names']),
            status='PRELIMINARY_8_OBSERVED_1_MODEL_ESTIMATED')
        report['runs'].append({'case':case,'samples':len(t),'eligible_start_points':int(valid.sum()),
                               'initial_interior_K':float(x[0,7]),'final_estimated_interior_K':float(x[-1,7])})
        print(f'T{case}: {len(t)} grid points; {valid.sum()} eligible starts; x8 estimated, excluded from target',flush=True)
    (args.output_dir/'experiment_definition.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved:',args.output_dir)


if __name__=='__main__': main()
