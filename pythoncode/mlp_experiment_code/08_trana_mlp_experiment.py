"""MLP + backpropagation genom COM; första experiment med delvis observerat tillstånd.

11 indata -> 32 Tanh -> 32 Tanh -> 11 gamma. Gamma hålls konstant i varje
prediktionsfönster. Träningsmålet är COM:s fria flerstegsprediktion av ÅTTA
uppmätta tillstånd. cabIntT är en separat modelluppskattning och saknar target.

Förträning approximerar den gamla parameterfunktionen. Därefter optimerar
backpropagation det verkliga prediktionsfelet genom den exporterade COM.
Detta är ett preliminärt identifieringsexperiment, inte en validerad regulator.
"""
from pathlib import Path
import argparse
import copy
import json
import hashlib
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from com_experiment_bridge import CasadiCOM, rollout

ROOT = Path(__file__).resolve().parents[2]
SCALE = np.array([1.,1.,1.,.01,1.,1e5,1e5,1.,1.])
MASK = np.array([True,True,True,True,True,True,True,False,True])
UNITS = ['K','K','K','fraction','K','Pa','Pa','unobserved','K']


class ExperimentMLP(nn.Module):
    def __init__(self, mean, std, lower, upper, nominal):
        super().__init__()
        for n,v in [('mean',mean),('std',std),('lower',lower),('upper',upper)]:
            self.register_buffer(n,torch.as_tensor(v,dtype=torch.float64).clone())
        self.net = nn.Sequential(nn.Linear(11,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,11)).double()
        nn.init.zeros_(self.net[-1].weight)
        span = np.asarray(upper)-np.asarray(lower)
        fraction = np.divide(np.asarray(nominal)-lower,span,out=np.full(11,.5),where=span>0)
        fraction = np.clip(fraction,1e-5,1-1e-5)
        with torch.no_grad():self.net[-1].bias.copy_(torch.tensor(np.log(fraction/(1-fraction))))

    def forward(self, features):
        # Om upper == lower hålls den parametern fast. p7 är nästan exakt
        # konstant i den befintliga tabellen och ges därför ingen fri variation.
        return self.lower+(self.upper-self.lower)*torch.sigmoid(self.net((features-self.mean)/self.std))


def features(run,k):
    return np.r_[float(run['T_amb_C']),run['x'][k],run['Ib_A'][k]]


def windows(runs, regions, horizon, stride=30):
    out=[]
    for r in runs:
        for start,stop in regions:
            for k in range(int(start),int(stop)-horizon+1,stride):
                if r['valid_start'][k] and k+horizon<len(r['t_s']):out.append((r,k))
    if not out:raise ValueError('Inga giltiga fönster')
    return out


def make_batch(windows,horizon,meta):
    ft=[];xx=[];us=[];ps=[];ys=[];prior=[]
    indices={n:i for i,n in enumerate(meta['fixed_parameter_names'])}
    # Endast givna EXTERNA storheter varierar med framtida tid.
    ext_names=['speed','batImeas','motQmeas','invQmeas','dcdcQmeas']
    for r,k in windows:
        ft.append(features(r,k));xx.append(r['x'][k]);us.append(r['u'][k:k+horizon])
        ys.append(r['x'][k+1:k+horizon+1]);prior.append(r['gamma_prior'][k])
        p=np.tile(r['fixed_at_start'][k],(horizon,1))
        for i,n in enumerate(ext_names):p[:,indices[n]]=r['exogenous'][k:k+horizon,i]
        # I identifieringen är den inspelade aktuatorsekvensen känd. Modellen
        # behåller sitt fan-filter med tau=5 s; detta är ingen MPC-prognosstudie.
        fan=float(r['u'][k,5]);alpha=np.exp(-1/5.)
        for j in range(horizon):
            fan=alpha*fan+(1-alpha)*r['u'][k+j,5]
            p[j,indices['fanRpm_prev']]=fan
        ps.append(p)
    return tuple(torch.as_tensor(np.stack(v),dtype=torch.float64)for v in (ft,xx,us,ps,ys,prior))


def loss_batch(model,engine,group,horizon,meta,baseline=False):
    f,x,u,p,y,gprior=make_batch(group,horizon,meta)
    gamma=gprior if baseline else model(f)
    pred=rollout(engine,x,u,p,gamma)
    error=(pred-y)/torch.tensor(SCALE)
    # Tillstånd 8 har uttryckligen INGEN target. Ingen framtida uppskattning
    # av inredningens temperatur används för att belöna nätverket.
    state_loss=error[:,:,MASK].square().mean()
    if not torch.isfinite(state_loss):raise FloatingPointError('Prediktionsfelet gav NaN/Inf')
    return state_loss,gamma,gprior,pred,y


def evaluate(model,engine,groups,horizon,meta,baseline=False):
    total=0.;squared=np.zeros(9);count=0
    model.eval()
    with torch.no_grad():
        for i in range(0,len(groups),16):
            b=groups[i:i+16]
            loss,_,_,pred,y=loss_batch(model,engine,b,horizon,meta,baseline)
            total+=float(loss)*len(b)
            squared+=np.square((pred-y).numpy()).sum(axis=(0,1));count+=len(b)*horizon
    rmse=np.sqrt(squared/count)
    return {'normalized_mse':total/len(groups),'rmse_by_state':
            [float(v)if MASK[j]else None for j,v in enumerate(rmse)],'windows':len(groups)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=ROOT/'Data'/'NN_training'/'experiment_8observed')
    ap.add_argument('--com-dir',type=Path,default=ROOT/'Data'/'NN_training'/'COM')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'Data'/'NN_training'/'mlp_experiment_results')
    ap.add_argument('--horizon',type=int,default=30)
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--pretrain-steps',type=int,default=600)
    ap.add_argument('--evaluate-test',action='store_true',help='Endast efter att modellvalet är klart: separat -7 C-utvärdering')
    args=ap.parse_args()
    if args.horizon not in range(1,31)or args.epochs<1:raise ValueError('Horisont 1..30, epochs >=1')
    torch.set_num_threads(1);torch.manual_seed(226);rng=np.random.default_rng(226)
    meta=json.loads((args.com_dir/'com_metadata.json').read_text(encoding='utf-8'))
    if meta['sample_time_s']!=1 or meta['rk4_substeps']!=1:raise ValueError('Väntade Ts=1, RK4 M=1')
    def read(case):
        r=dict(np.load(args.data_dir/f'experiment_T{case}.npz',allow_pickle=False))
        n=len(r['t_s'])
        expected={'x':(n,9),'u':(n-1,6),'fixed_at_start':(n,63),'exogenous':(n,5),'gamma_prior':(n,11)}
        for key,shape in expected.items():
            if r[key].shape!=shape:raise ValueError(f'Fel form för T{case}/{key}: {r[key].shape}')
        for key in ['x','u','exogenous','gamma_prior']:
            if not np.isfinite(r[key]).all():raise ValueError(f'T{case}/{key}: NaN/Inf')
        if not np.isfinite(r['fixed_at_start'][r['valid_start']]).all():raise ValueError('Ogiltiga fasta parametrar')
        if float(r['T_amb_C'])!=-case:raise ValueError('Fel omgivningstemperatur')
        if not np.array_equal(r['target_mask'],MASK):raise ValueError('Fel target-mask')
        for key in ['state_names','control_names','fixed_parameter_names']:
            if list(r[key])!=meta[key]:raise ValueError(f'Fel ordning: {key}')
        if not np.allclose(np.diff(r['t_s']),1)or r['t_s'][0]!=0:raise ValueError('Fel tidsgrid')
        return r
    runs=[read(5),read(10)]
    # Validera både en del av körningen med körlast och den sena parkerade fasen.
    # Inga överlappande fönster över gränser; minst 30 s mellan tidsblocken.
    # Detta är fortfarande validering INOM samma körcykel. Den modellbaserade
    # x8-prediktorn använder kausalt tillgänglig mäthistorik genom hela loggen.
    train=windows(runs,[(30,900),(1200,2700)],args.horizon)
    val_drive=windows(runs,[(930,1170)],args.horizon)
    val_park=windows(runs,[(2730,3600)],args.horizon)
    ft=np.stack([features(r,k)for r,k in train]);gp=np.stack([r['gamma_prior'][k]for r,k in train])
    # Rent empiriska försöksgränser från TRAIN-delens befintliga gamma-prior.
    # Ingen fysisk säkerhetsgaranti. Test-/valideringsdata sätter inga gränser.
    lo,hi=gp.min(0),gp.max(0);span=hi-lo;constant=span<1e-6
    lower=np.maximum(.05,lo-.1*span);upper=hi+.1*span
    lower[constant]=upper[constant]=np.median(gp[:,constant],axis=0)
    nominal=np.median(gp,axis=0)
    model=ExperimentMLP(ft.mean(0),np.maximum(ft.std(0),1e-8),lower,upper,nominal)
    engine=CasadiCOM(args.com_dir/'vehicle_com.casadi')
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    ftorch=torch.tensor(ft);gtorch=torch.tensor(gp)
    gscale=torch.tensor(np.where(constant,1.,upper-lower))
    # Starta nära befintlig parameterfunktion. Denna fas är en approximation
    # av en gammal modell, inte bevis för förbättrad fordonsprediktion.
    for step in range(args.pretrain_steps):
        ids=rng.integers(0,len(ft),size=min(64,len(ft)))
        opt.zero_grad();loss=((model(ftorch[ids])-gtorch[ids])/gscale).square().mean()
        loss.backward();opt.step()
    print(f'Pretraining complete: {args.pretrain_steps} steps; gamma approximation only',flush=True)
    bd=evaluate(model,engine,val_drive,args.horizon,meta,True)
    bp=evaluate(model,engine,val_park,args.horizon,meta,True)
    initial_d=evaluate(model,engine,val_drive,args.horizon,meta)
    initial_p=evaluate(model,engine,val_park,args.horizon,meta)
    def score(d,p):return .5*(d['normalized_mse']+p['normalized_mse'])
    # Ta även med startmodellen som kandidat, så att en försämrad optimering
    # inte råkar ersätta en bättre förtränad modell.
    best=score(initial_d,initial_p);best_state=copy.deepcopy(model.state_dict());best_epoch=0
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    history=[];bad=0
    print(f'Baseline validation score: {score(bd,bp):.6g}; pretrained MLP: {best:.6g}',flush=True)
    for epoch in range(args.epochs):
        model.train();order=rng.permutation(len(train));total=0.
        for start in range(0,len(train),16):
            group=[train[i]for i in order[start:start+16]]
            opt.zero_grad();state,g,gprior,_,_=loss_batch(model,engine,group,args.horizon,meta)
            # Svag prior reglerar dåligt identifierade gamma, huvudmålet är states.
            penalty=((g-gprior)/gscale).square().mean()
            objective=state+1e-3*penalty;objective.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all()for p in model.parameters()):
                raise FloatingPointError('Ogiltig nätverksgradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();total+=float(objective.detach())*len(group)
        vd=evaluate(model,engine,val_drive,args.horizon,meta)
        vp=evaluate(model,engine,val_park,args.horizon,meta)
        selected=score(vd,vp)
        history.append({'epoch':epoch+1,'train_objective':total/len(train),
                        'val_drive_mse':vd['normalized_mse'],'val_park_mse':vp['normalized_mse'],'val_score':selected})
        print(f'Epoch {epoch+1}: train={total/len(train):.6g}, val={selected:.6g}, interpolation={score(bd,bp):.6g}',flush=True)
        if selected<best:
            best=selected;best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict());bad=0
        else:bad+=1
        if bad>=8:break
    model.load_state_dict(best_state)
    final_drive=evaluate(model,engine,val_drive,args.horizon,meta)
    final_park=evaluate(model,engine,val_park,args.horizon,meta)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result={'status':'PRELIMINARY_PARTIAL_OBSERVATION_EXPERIMENT',
        'horizon_s':args.horizon,'rk4_substeps':1,'seed':226,'train_windows':len(train),
        'train_time_blocks_s':[[30,900],[1200,2700]],
        'val_time_blocks_s':[[930,1170],[2730,3600]],
        'validation_score_definition':'0.5 driving normalized MSE + 0.5 parked normalized MSE',
        'best_epoch':best_epoch,'pretrain_steps':args.pretrain_steps,
        'state_names':meta['state_names'],'state_rmse_units':UNITS,'state_error_scale':SCALE.tolist(),
        'target_mask':MASK.tolist(),'latent_state':'causal fixed-prior cabin interior prediction, not measured',
        'empirical_gamma_lower':lower.tolist(),'empirical_gamma_upper':upper.tolist(),
        'held_constant_gamma':[int(i+1) for i in np.flatnonzero(constant)],
        'pretrained_validation_score':score(initial_d,initial_p),'final_validation_score':best,
        'interpolation_validation_score':score(bd,bp),
        'validation_drive_mlp':final_drive,'validation_drive_interpolation':bd,
        'validation_park_mlp':final_park,'validation_park_interpolation':bp,
        'history':history,'test_evaluated':False,'independent_drive_cycle_test':False,
        'thermo_matlab_parity_verified':False,'interpolation_matlab_parity_verified':False,
        'known_future_inputs_used_for_identification':True,
        'com_sha256':hashlib.sha256((args.com_dir/'vehicle_com.casadi').read_bytes()).hexdigest()}
    if args.evaluate_test:
        test=read(7);test_windows=windows([test],[(30,3600)],args.horizon)
        result.update(test_evaluated=True,test_mlp=evaluate(model,engine,test_windows,args.horizon,meta),
                      test_interpolation=evaluate(model,engine,test_windows,args.horizon,meta,True))
    torch.save({'model_state':best_state,'result':result,'metadata':meta},args.output_dir/'experimental_mlp.pt')
    (args.output_dir/'training_metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    ax.plot([0]+[h['epoch']for h in history],[score(initial_d,initial_p)]+[h['val_score']for h in history],label='MLP + COM, validation')
    ax.axhline(score(bd,bp),linestyle='--',color='black',label='Existing gamma interpolation (Python reproduction)')
    ax.set(xlabel='Epoch',ylabel='Scaled state prediction MSE',title=f'Preliminary training — {args.horizon}-second COM rollout')
    ax.grid(alpha=.4);ax.legend();fig.savefig(args.output_dir/'training_validation.png',dpi=150);plt.close(fig)
    # Ett exempel på fri simulering. Referenstillstånd återställs endast i
    # början av fönstret. Inredningstillståndet visas inte som uppmätt facit.
    group=val_drive[:1]
    with torch.no_grad():
        _,_,_,pr,y=loss_batch(model,engine,group,args.horizon,meta)
        _,_,_,br,_=loss_batch(model,engine,group,args.horizon,meta,True)
    ts=group[0][0]['t_s'][group[0][1]+1:group[0][1]+args.horizon+1]
    fig,axs=plt.subplots(4,2,figsize=(13,11),layout='constrained')
    titles=['Motor','Inverter','DC/DC converter','Battery SOC','Battery','Compressor inlet','Compressor outlet','Cabin air']
    for ax,j,title in zip(axs.flat,np.flatnonzero(MASK),titles):
        factor=1e-5 if j in(5,6)else 100. if j==3 else 1.
        offset=0. if j in(3,5,6)else 273.15
        for arr,label,style in [(y,'Measured','-'),(br,'Prior COM','--'),(pr,'MLP + COM','-.')]:
            ax.plot(ts,(arr.numpy()[0,:,j]-offset)*factor,style,label=label)
        ax.set(title=title,xlabel='Time [s]',ylabel='bar'if j in(5,6)else '%'if j==3 else '°C');ax.grid(alpha=.35)
    axs.flat[0].legend();fig.savefig(args.output_dir/'validation_rollout_example.png',dpi=150);plt.close(fig)
    print('Saved experimental model, metrics and figures:',args.output_dir,flush=True)
    print('Test -7 C evaluated:',result['test_evaluated'])


if __name__=='__main__':main()
