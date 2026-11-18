"""Gradientkontroll genom ER exporterade vehicle_com.casadi.

Steg 03 testade bryggan med ett syntetiskt F. Detta program testar i stället
fordonets exporterade F genom tre RK4-steg. Det är ett PROGRAMVARUTEST,
inte en jämförelse mot fordonets mätdata och inte en tränad modell.

CoolProp används för en direkt Python-översättning av de packade fälten i
estimateParameters.m. Den översättningen är ännu inte jämförd numeriskt
med MATLAB. Därför används den här bara för att skapa ett testfall.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import casadi as ca
import torch
from scipy.io import loadmat, savemat
from com_experiment_bridge import CasadiCOM, rollout
from thermo_properties import parameters

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--com-dir', type=Path, default=ROOT/'Data'/'NN_training'/'COM')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'reports'/'context_validation')
    parser.add_argument('--tuning-file', type=Path,
                        help='Valfritt: använd medianen i befintlig parametertabell som testpunkt')
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata = json.loads((args.com_dir/'com_metadata.json').read_text(encoding='utf-8'))
    if metadata['sample_time_s'] != 1 or metadata['rk4_substeps'] != 1:
        raise ValueError('Testet väntar Ts=1 s, RK4 M=1')
    f = ca.Function.load(str(args.com_dir/'vehicle_com.casadi'))
    if [f.size1_in(i) for i in range(3)] != [9,6,74]:
        raise ValueError('Fel funktionsdimensioner')
    # Kontrollerad testpunkt. Temperaturer motsvarar kallstart vid -5 C.
    # Aktuatorerna väljs för att aktivera modellens olinjära samband. De är
    # INTE påstådda mätningar vid detta tillstånd. Inga mål från körningen används.
    x = np.full(9, 268.15); x[3] = 1.; x[5:7] = 250000.
    u = np.array([500., .1, 600., 600., 2000., 100.])
    gamma = np.array([1.,1.,1.,1.,1.,2.375,5.05,.925,1.,1.,1.])
    if args.tuning_file:
        tv = loadmat(args.tuning_file, simplify_cells=True)['tunedVariables']
        gamma = np.median(np.asarray(tv['parameters'], float), axis=0)
    fields = parameters(x, -5., ctrl=(1.,1.,1.,0.,0.,2.,1.))
    fixed = np.array([fields[n] for n in metadata['fixed_parameter_names']])
    engine = CasadiCOM(args.com_dir/'vehicle_com.casadi', sample_time=1.)
    horizon = 3
    ut = torch.tensor(np.tile(u, (1,horizon,1)), dtype=torch.float64)
    pt = torch.tensor(np.tile(fixed, (1,horizon,1)), dtype=torch.float64)
    # Normalisera testkoordinaterna. Då blir differensstegen meningsfulla
    # även för Pa, K och SOC som har helt olika storleksordningar.
    sx = torch.tensor([20.,20.,20.,.1,20.,1e5,1e6,20.,20.], dtype=torch.float64)
    sg = torch.tensor(np.maximum(np.abs(gamma), .1), dtype=torch.float64)
    x0 = torch.tensor(x, dtype=torch.float64)
    g0 = torch.tensor(gamma, dtype=torch.float64)
    z = torch.zeros((1,20), dtype=torch.float64, requires_grad=True)
    def composed(z):
        state = x0 + sx*z[:,:9]
        g = g0 + sg*z[:,9:]
        return (rollout(engine, state, ut, pt, g)-x0)/sx
    result = torch.autograd.gradcheck(composed, (z,), eps=1e-5, atol=2e-5, rtol=2e-3,
                                     raise_exception=True, fast_mode=False)
    prediction = composed(z)
    derivative = torch.autograd.functional.jacobian(composed, z)
    if not torch.isfinite(derivative).all():
        raise FloatingPointError('Gradienten är inte ändlig')
    args.report_dir.mkdir(parents=True, exist_ok=True)
    # MATLAB kan läsa detta testfall och jämföra samma serialiserade F.
    p = np.r_[fixed, gamma]
    outputs = f(x,u,p)
    fixture = {'x': x[:,None], 'u':u[:,None], 'p':p[:,None],
               'xdot_python':np.array(outputs[0]), 'outputs_python':np.array(outputs[1]),
               'power_python':np.array(outputs[2])}
    savemat(args.report_dir/'com_numeric_fixture.mat', fixture)
    report = {
        'function': 'vehicle_com.casadi', 'input_dimensions': [9,6,74],
        'gradient_check': bool(result), 'horizon': horizon, 'sample_time_s':1.,
        'integration': 'RK4 M=1', 'dtype':'float64',
        'finite_difference_eps_dimensionless':1e-5,'atol':2e-5,'rtol':2e-3,
        'jacobian_all_finite':True,
        'casadi_version': ca.__version__, 'torch_version':torch.__version__,
        'thermo_python_vs_matlab_verified':False,
        'numeric_matlab_equivalence_verified':False,
        'validated_against_vehicle_measurements':False,
        'training_performed':False,
        'fixture': 'controlled numerical test, not a recorded trajectory',
    }
    (args.report_dir/'actual_com_gradient_check.json').write_text(json.dumps(report, indent=2),encoding='utf-8')
    print('PASS: numerical gradient check through actual vehicle COM, 3 RK4 steps')
    print('PASS: gradients with respect to initial states and all 11 gamma inputs')
    print('Software test only; vehicle prediction accuracy and MATLAB thermo parity not established.')
    print('Saved:', args.report_dir)


if __name__ == '__main__':
    main()
