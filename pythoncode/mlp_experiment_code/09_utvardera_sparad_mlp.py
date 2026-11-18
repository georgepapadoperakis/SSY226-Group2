"""Separat test av den SPARADE modellen vid -7 C, utan omträning.

Kör först när modellval och inställningar är låsta. Denna fil har avsiktligt
inte körts i den medföljande leveransen. -7 C ligger mellan träningstemperaturerna
-5 och -10 C; det är temperaturinterpolation inom samma typ av körcykel.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'Data'/'NN_training'/'mlp_experiment_results'/'experimental_mlp.pt')
    ap.add_argument('--data-dir',type=Path,default=ROOT/'Data'/'NN_training'/'experiment_8observed')
    ap.add_argument('--com-dir',type=Path,default=ROOT/'Data'/'NN_training'/'COM')
    args=ap.parse_args()
    spec=importlib.util.spec_from_file_location('mlp_experiment',Path(__file__).with_name('08_trana_mlp_experiment.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    result=saved['result'];state=saved['model_state'];meta=saved['metadata']
    lower=np.array(result['empirical_gamma_lower']);upper=np.array(result['empirical_gamma_upper'])
    model=module.ExperimentMLP(state['mean'],state['std'],lower,upper,(lower+upper)/2)
    model.load_state_dict(state);model.eval();torch.set_num_threads(1)
    # Kontrollera att testets COM är samma som vid träningen.
    import hashlib
    if hashlib.sha256((args.com_dir/'vehicle_com.casadi').read_bytes()).hexdigest()!=result['com_sha256']:
        raise ValueError('COM-filen har ändrats sedan träningen')
    run=dict(np.load(args.data_dir/'experiment_T7.npz',allow_pickle=False))
    if float(run['T_amb_C'])!=-7. or not np.array_equal(run['target_mask'],module.MASK):
        raise ValueError('Fel testfall eller target-mask')
    for key in ['state_names','control_names','fixed_parameter_names']:
        if list(run[key])!=meta[key]:raise ValueError('Fel signalordning')
    horizon=int(result['horizon_s'])
    groups=module.windows([run],[(30,3600)],horizon)
    engine=module.CasadiCOM(args.com_dir/'vehicle_com.casadi')
    metrics={'test_case_C':-7.,'horizon_s':horizon,'model_retrained':False,
             'interpolation_test':True,'independent_drive_cycle_test':False,
             'state_names':meta['state_names'],
             'mlp':module.evaluate(model,engine,groups,horizon,meta),
             'prior_com':module.evaluate(model,engine,groups,horizon,meta,True)}
    path=args.checkpoint.with_name('test_T7_metrics.json')
    path.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('Saved held-out temperature test:',path)
    print('MLP normalized MSE:',metrics['mlp']['normalized_mse'])
    print('Prior COM normalized MSE:',metrics['prior_com']['normalized_mse'])
    print('cabIntT is model-estimated and has no target. This is not an independent drive-cycle test.')


if __name__=='__main__':main()
