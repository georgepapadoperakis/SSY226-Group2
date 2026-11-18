"""Visa MLP-resultaten över hela körningen, utan att träna om nätverket.

Spara i projektroten SSY226-Group2 och kör i din befintliga .venv:
    python visa_mlp_hela_forloppet.py --show

Två utvärderingar hålls tydligt isär:
1) continuous: en sammanhängande simulering från t=30 s; inga tillstånd återställs
   från mätdata senare. Gamma och termodynamiska egenskaper uppdateras varje
   sekund FRÅN MODELLENS EGNA tillstånd. Det är ett nytt långt stresstest;
   vid träningen hölls gamma/egenskaper under ett 30-sekundersfönster.
2) reset: samma korta prediktionsupplägg som träningen, men över hela tidsaxeln.
   Tillstånd återställs från loggen vid början av varje fönster. Detta är INTE
   en sammanhängande simulering och markeras uttryckligen i bild och CSV.

Dina inspelade aktuatorer och externa störningar används i båda fallen.
Detta är ingen sluten MPC-simulering. Ett modelluppskattat inredningstillstånd
finns internt men används aldrig som uppmätt facit. Uppstart före 30 s utelämnas.
-7 C-testet används inte. Standardfallet är -10 C, som i kollegornas figur.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

STATE_IDS = [0,1,2,3,4,5,6,8]
NAMES = {0:'Motor temperature',1:'Inverter temperature',2:'DC/DC converter temperature',
         3:'Battery state of charge',4:'Battery temperature',5:'Low-side pressure',
         6:'High-side pressure',8:'Cabin air temperature'}
DISPLAY_UNITS = {j:('bar' if j in (5,6) else 'percentage points' if j==3 else 'degC') for j in STATE_IDS}


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def display(values,j):
    """Omvandla till läsbara enheter. Temperaturfel i K och degC är lika stora."""
    y=np.asarray(values)
    return y/1e5 if j in(5,6) else y*100 if j==3 else y-273.15


def simulate_continuous(run,model,step,prior,thermo,meta,pcrit,start):
    t=run['t_s'];n=len(t);ambient=float(run['T_amb_C'])
    predictions={};statuses={}
    for label in ('prior','mlp'):
        # Efter denna rad tilldelas x ALDRIG något framtida referenstillstånd.
        # Endast givna aktuatorer och externa störningar hämtas ur loggen.
        x=run['x'][start].copy()
        pred=np.full((n,9),np.nan);pred[start]=x
        fan=float(run['u'][start,5]);alpha=np.exp(-1/5.)
        status={'completed':False,'stop_reason':None,'last_recorded_s':float(t[start])}
        print(f'Continuous {label}: start t={t[start]:g} s',flush=True)
        for k in range(start,n-1):
            try:
                if not np.isfinite(x).all():raise ValueError('Simulated state contains NaN/Inf')
                if min(x[5:7])<=100:raise ValueError('Simulated pressure <=100 Pa: saturation derivatives unavailable')
                if max(x[5:7])+100>=pcrit:
                    raise ValueError('Simulated pressure reaches R134a critical pressure: saturation-property model unavailable')
                u=run['u'][k]
                ext=run['exogenous'][k] # km/h, positiv urladdningsström, tre värmekällor
                fan=alpha*fan+(1-alpha)*u[5]
                # Samma kalla logikantagande som i träningsförberedelsen.
                fields=thermo(x,ambient,speed=ext[0],ib=ext[1],heat=ext[2:5],
                              fan=fan,ctrl=(0.,1.,1.,0.,0.,2.,1.))
                fixed=np.array([fields[name]for name in meta['fixed_parameter_names']],dtype=float)
                if label=='prior':gamma=prior(x[None,:],ambient)[0]
                else:
                    feature=np.r_[ambient,x,run['Ib_A'][k]]
                    with torch.no_grad():gamma=model(torch.tensor(feature[None,:],dtype=torch.float64)).numpy()[0]
                if not np.isfinite(fixed).all()or not np.isfinite(gamma).all():
                    raise ValueError('Invalid thermodynamic fields or gamma')
                x=np.array(step(x,u,fixed,gamma)).reshape(9)
                if not np.isfinite(x).all():raise ValueError('COM step returned NaN/Inf')
                pred[k+1]=x;status['last_recorded_s']=float(t[k+1])
            except (ValueError,RuntimeError,FloatingPointError)as exc:
                status['stop_reason']=str(exc)
                status['failure_time_s']=float(t[k])
                print(f'STOP continuous {label} at {t[k]:g} s: {exc}',flush=True)
                break
            if (k+1)%600==0:print(f'  {label}: reached {t[k+1]:g} s',flush=True)
        else:status['completed']=True
        status['last_recorded_pressure_bar']=(pred[np.isfinite(pred).all(axis=1)][-1,5:7]/1e5).tolist()
        predictions[label]=pred;statuses[label]=status
    return predictions,statuses


def simulate_resets(run,model,engine,module,meta,start,horizon):
    n=len(run['t_s']);predictions={name:np.full((n,9),np.nan)for name in ('prior','mlp')}
    groups=[]
    for k in range(start,n-horizon,horizon):
        if run['valid_start'][k]:groups.append((run,k))
    # range måste också inkludera ett fönster som slutar exakt vid sista provet.
    final=n-1-horizon
    if final>=start and (final-start)%horizon==0 and run['valid_start'][final]:
        if not groups or groups[-1][1]!=final:groups.append((run,final))
    # Samma batchfunktion, gamma-hållning och fan-prognos som vid träningen.
    with torch.no_grad():
        for i in range(0,len(groups),16):
            batch=groups[i:i+16]
            for name,baseline in [('prior',True),('mlp',False)]:
                _,_,_,pr,_=module.loss_batch(model,engine,batch,horizon,meta,baseline)
                pr=pr.numpy()
                for row,(_,k)in enumerate(batch):predictions[name][k+1:k+horizon+1]=pr[row]
    return predictions,[k for _,k in groups]


def score_and_export(root,case,protocol,run,predictions,statuses,result):
    """Bedöm båda modellerna över exakt samma tidsstämplar.

    Om en fri simulering stannar jämförs endast det gemensamma beräknade
    intervallet. Täckningen och stopporsaken rapporteras alltid bredvid felet.
    Ett kortare lyckat prefix får inte beskrivas som en fullständig validering.
    """
    t=run['t_s'];ref=run['x']
    common=np.isfinite(predictions['prior'][:,STATE_IDS]).all(axis=1)&np.isfinite(predictions['mlp'][:,STATE_IDS]).all(axis=1)
    common &= t>30 # initialpunkten är ingen prediktion
    scopes={'all_available_common_times':common}
    if protocol=='reset':
        # Samma två avskilda valideringsblock som vid modellvalet.
        val=((t>930)&(t<=1170))|((t>2730)&(t<=3600))
        scopes['validation_times_only']=common&val
    rows=[];summary={}
    for scope,mask in scopes.items():
        summary[scope]={'points':int(mask.sum()),'fraction_of_prediction_grid':float(mask.sum()/max(1,(t>30).sum())),
                        'first_time_s':float(t[mask][0])if mask.any()else None,
                        'last_time_s':float(t[mask][-1])if mask.any()else None}
        for label in ('prior','mlp'):
            for j in STATE_IDS:
                error=display(predictions[label][mask,j],j)-display(ref[mask,j],j)
                rows.append({'protocol':protocol,'scope':scope,'model':label,
                             'state':str(run['state_names'][j]),'unit':DISPLAY_UNITS[j],
                             'samples':len(error),
                             'RMSE':float(np.sqrt(np.mean(error**2)))if len(error)else None,
                             'MAE':float(np.mean(np.abs(error)))if len(error)else None,
                             'max_abs_error':float(np.max(np.abs(error)))if len(error)else None})
    path=root/f'T{case}_{protocol}_metrics.csv'
    with path.open('w',newline='',encoding='utf-8-sig')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Fulla tidsserier i visningsenheter för Excel/MATLAB/rapport.
    display_names={0:'motor_degC',1:'inverter_degC',2:'dcdc_degC',3:'SOC_percent',
                   4:'battery_degC',5:'compressor_inlet_bar',6:'compressor_outlet_bar',8:'cabin_air_degC'}
    fields=['time_s','data_partition']
    for label in ('reference','prior','mlp'):
        fields += [f'{label}_{display_names[j]}'for j in STATE_IDS]
    with (root/f'T{case}_{protocol}_timeseries.csv').open('w',newline='',encoding='utf-8-sig')as f:
        writer=csv.writer(f);writer.writerow(fields)
        arrays={'reference':ref,**predictions}
        transformed={label:{j:display(a[:,j],j)for j in STATE_IDS}for label,a in arrays.items()}
        for k,time in enumerate(t):
            partition=('excluded_startup'if time<30 else
                       'validation_drive'if 930<time<=1170 else
                       'validation_park'if 2730<time<=3600 else
                       'training_region'if (30<=time<=900 or 1200<=time<=2700)else 'gap')
            writer.writerow([time,partition]+[transformed[label][j][k]if np.isfinite(transformed[label][j][k])else ''
                                             for label in ('reference','prior','mlp')for j in STATE_IDS])
    np.savez_compressed(root/f'T{case}_{protocol}_predictions.npz',t_s=t,reference=ref,
                        prior=predictions['prior'],mlp=predictions['mlp'],
                        state_names=run['state_names'],observed_state_mask=module_mask())
    return {'coverage':summary,'statuses':statuses,'metrics_file':path.name,'rmse_rows':rows}


def module_mask():return np.array([1,1,1,1,1,1,1,0,1],bool)


def plot_sheets(output,case,protocol,run,predictions,statuses,reset_starts,horizon,show):
    t=run['t_s'];protocol_title=('Continuous simulation — no state resets; gamma/properties updated every 1 s'
                                if protocol=='continuous'else f'{horizon}-second predictions — states reset from log every {horizon} s')
    # Det första bladet använder samma fyra variabler som kollegornas bild.
    sheets=[('main',[8,4,5,6]),('powertrain',[0,1,2,3])]
    for sheet,indices in sheets:
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        for ax,j in zip(axes.flat,indices):
            ax.plot(t,display(run['x'][:,j],j),'k--',lw=1.5,label='Logged reference')
            for label,color,title in [('prior','tab:red','COM, prior gamma interpolation'),('mlp','tab:blue','COM, MLP gamma')]:
                y=display(predictions[label][:,j],j).copy()
                if protocol=='reset':
                    # Bryt linjen vid omstarterna, så att sammanfogade korta
                    # prediktioner inte ser ut som en fri sammanhängande kurva.
                    y[np.array(reset_starts[1:],dtype=int)]=np.nan
                ax.plot(t,y,color=color,lw=1.25,label=title)
            ax.axvspan(930,1170,color='tab:green',alpha=.07)
            ax.axvspan(2730,3600,color='tab:green',alpha=.07)
            ax.set(title=NAMES[j],xlabel='Time [s]',ylabel='Pressure [bar]'if j in(5,6)else 'SOC [%]'if j==3 else 'Temperature [°C]')
            ax.grid(alpha=.35);ax.legend(fontsize=8)
        short='; '.join(f'{name} stopped at {s["last_recorded_s"]:g} s'for name,s in statuses.items()if not s.get('completed',True))
        fig.suptitle(f'Ambient −{case} °C — {protocol_title}\nStarts at 30 s; green shading = validation blocks'+('\n'+short if short else ''),fontsize=12)
        fig.savefig(output/f'T{case}_{protocol}_{sheet}.png',dpi=160)
        if not show:plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',type=Path)
    ap.add_argument('--case',type=int,choices=[5,10],default=10,help='Börja med -10 C; -7 C-testet berörs inte')
    ap.add_argument('--tuning-file',type=Path)
    ap.add_argument('--show',action='store_true',help='Visa de fyra diagramfönstren efter beräkningen')
    args=ap.parse_args()
    here=Path(__file__).resolve().parent
    project=args.project_root.resolve()if args.project_root else here.parent if here.name.lower()=='pythoncode'else here
    code=project/'pythoncode'/'mlp_experiment_code'
    if not (code/'08_trana_mlp_experiment.py').exists():code=project/'pythoncode'
    if not (code/'08_trana_mlp_experiment.py').exists():
        raise FileNotFoundError('Hittar inte den utskrivna träningskoden. Spara visningsfilen i SSY226-Group2.')
    sys.path.insert(0,str(code))
    module=import_file('mlp_plot_training',code/'08_trana_mlp_experiment.py')
    prep=import_file('mlp_plot_preparation',code/'07_forbered_observerat_mlp_experiment.py')
    from thermo_properties import parameters
    from CoolProp.CoolProp import PropsSI
    import casadi as ca
    torch.set_num_threads(1)
    base=project/'Data'/'NN_training'
    saved=torch.load(base/'mlp_experiment_results'/'experimental_mlp.pt',map_location='cpu',weights_only=True)
    meta=saved['metadata'];result=saved['result'];state=saved['model_state']
    function_file=base/'COM'/'vehicle_com.casadi'
    if hashlib.sha256(function_file.read_bytes()).hexdigest()!=result['com_sha256']:
        raise ValueError('COM-filen är en annan version än den som användes vid träningen')
    lower=np.array(result['empirical_gamma_lower']);upper=np.array(result['empirical_gamma_upper'])
    model=module.ExperimentMLP(state['mean'],state['std'],lower,upper,(lower+upper)/2)
    model.load_state_dict(state);model.eval()
    run=dict(np.load(base/'experiment_8observed'/f'experiment_T{args.case}.npz',allow_pickle=False))
    for name in ('state_names','control_names','fixed_parameter_names'):
        if list(run[name])!=meta[name]:raise ValueError(f'Fel signalordning för {name}')
    if not np.isclose(float(run['T_amb_C']),-args.case):raise ValueError('Fel temperatur i datafilen')
    if not np.array_equal(run['target_mask'],module_mask()):raise ValueError('Fel target-mask')
    if args.tuning_file is None:
        matches=sorted(project.rglob('tunedVariables_minus5_minus10.mat'),key=lambda p:len(p.parts))
        if not matches:raise FileNotFoundError('Ange --tuning-file till den befintliga parametertabellen')
        args.tuning_file=matches[0]
    definition_path=base/'experiment_8observed'/'experiment_definition.json'
    if definition_path.exists():
        definition=json.loads(definition_path.read_text(encoding='utf-8'))
        expected_table=definition.get('table_sha256')
        if expected_table and hashlib.sha256(args.tuning_file.read_bytes()).hexdigest()!=expected_table:
            raise ValueError('Parametertabellen skiljer sig från den som användes vid förberedelsen')
    prior=prep.GammaInterpolation(args.tuning_file)
    engine=module.CasadiCOM(function_file)
    # Endast värdet behövs i efterutvärderingen, inte alla Jacobianer.
    sx=ca.SX.sym('x',9);su=ca.SX.sym('u',6);sp=ca.SX.sym('p',63);sg=ca.SX.sym('g',11)
    step=ca.Function('plot_value_only',[sx,su,sp,sg],[engine.evaluate(sx,su,sp,sg)[0]])
    start=int(np.flatnonzero(np.isclose(run['t_s'],30.))[0])
    output=base/'mlp_full_evaluation';output.mkdir(parents=True,exist_ok=True)
    horizon=int(result['horizon_s'])
    print(f'Evaluating saved model at -{args.case} C; no retraining; -7 C untouched.',flush=True)
    full,status=simulate_continuous(run,model,step,prior,parameters,meta,PropsSI('pcrit','R134a'),start)
    full_summary=score_and_export(output,args.case,'continuous',run,full,status,result)
    plot_sheets(output,args.case,'continuous',run,full,status,[],horizon,args.show)
    reset,starts=simulate_resets(run,model,engine,module,meta,start,horizon)
    reset_status={label:{'completed':True,'protocol':'state reset at each window start'}for label in ('prior','mlp')}
    reset_summary=score_and_export(output,args.case,'reset',run,reset,reset_status,result)
    plot_sheets(output,args.case,'reset',run,reset,reset_status,starts,horizon,args.show)
    report={'ambient_C':-args.case,'is_independent_test':False,'case_was_a_training_temperature':True,
            'startup_excluded_before_s':30.,'continuous':full_summary,'reset':reset_summary,
            'continuous_gamma_property_update_s':1,'training_horizon_s':horizon,
            'recorded_actuators_and_external_disturbances_used':True,
            'latent_interior_has_no_measurement_target':True,'matlab_parity_verified':False}
    (output/f'T{args.case}_evaluation_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved diagrams, numerical time series and RMSE/MAE tables:',output,flush=True)
    for protocol,summary in [('continuous',full_summary),('reset',reset_summary)]:
        c=summary['coverage']['all_available_common_times']
        print(f'{protocol}: common prediction coverage {c["first_time_s"]}..{c["last_time_s"]} s; {c["points"]} points',flush=True)
    if args.show:plt.show()


if __name__=='__main__':main()
