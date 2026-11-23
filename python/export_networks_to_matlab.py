"""
Export the two trained γ networks to .mat files for MATLAB (Source Files/gamma_network.m reads them).

    models/gamma11_original_com.pt  ->  models/gamma11_original_com.mat   (used when COM_MODEL = 'nn11')
    models/gamma15_fixed_com.pt     ->  models/gamma15_fixed_com.mat      (used when COM_MODEL = 'nn15')

Each .mat also holds test cases (inputs + the γ PyTorch gives for them): in MATLAB, `gamma_network()` without
arguments recomputes them and prints PASS when MATLAB and PyTorch agree (to about 1e-15).
Run from the python folder after retraining:   python export_networks_to_matlab.py
"""
import numpy as np
import scipy.io as sio
import torch
from gamma_freerun import *

U_SCALE = np.array([5000, 0.2, 1200, 1200, 4000, 1000.0])      # same as features() in gamma_net.py
Q_SCALE = np.array([4300, 240, 175.0])


def test_windows(n_per_run=25, seed=0):
    """Deterministic sample of 30-s windows from the 8 test runs (+ start-up windows): inputs for the self-test."""
    PB = load_profiles(TEST_KEYS)
    rng = np.random.default_rng(seed)
    ws = []
    for k in TEST_KEYS:
        w = make_windows(PB[k], H=30, stride=30, startup_stride=3)
        idx = np.sort(rng.choice(len(w["k0"]), size=min(n_per_run, len(w["k0"])), replace=False))
        ws.append(subset(w, torch.as_tensor(idx)))
    return cat_windows(ws)


def export(net, mu, sd, path, W):
    sdict = {k: v.detach().double().numpy() for k, v in net.state_dict().items()}
    z = features(W)
    with torch.no_grad():
        g = net((z - mu) / sd).double().numpy()
    base = net if isinstance(net, GammaNet) else net.net12.base
    out = dict(W1=sdict[pre(net) + "net.0.weight"], b1=sdict[pre(net) + "net.0.bias"],
               W2=sdict[pre(net) + "net.2.weight"], b2=sdict[pre(net) + "net.2.bias"],
               W3=sdict[pre(net) + "net.4.weight"], b3=sdict[pre(net) + "net.4.bias"],
               mu=np.asarray(mu, float).ravel(), sd=np.asarray(sd, float).ravel(),
               log_lo=float(base.log_lo), log_hi=float(base.log_hi), u_scale=U_SCALE, q_scale=Q_SCALE,
               test_Tamb=W["Tamb"].numpy(), test_x=W["x0"].numpy(), test_Uplan=W["u"].numpy(),
               test_motQ=W["est"]["motQmeas"][:, 0].numpy(), test_invQ=W["est"]["invQmeas"][:, 0].numpy(),
               test_batI=W["est"]["batImeas"][:, 0].numpy(), test_features=z.numpy(), test_gamma=g)
    if not isinstance(net, GammaNet):                       # 15-γ network: γ12 head and γ13..γ15 head
        out.update(W12=sdict["net12.head12.weight"], b12=sdict["net12.head12.bias"], g12_max=GammaNet12.G12_MAX,
                   Wh=sdict["head.weight"], bh=sdict["head.bias"])
    out["about"] = (f"{path.name}: γ network for MATLAB (Source Files/gamma_network.m). {g.shape[1]} outputs. "
                    f"Made by python/export_networks_to_matlab.py from {path.with_suffix('.pt').name}.")
    sio.savemat(path, out, do_compression=True)
    print(f"saved {path}  ({g.shape[1]} γ, {len(g)} test cases)")


def pre(net):
    return "" if isinstance(net, GammaNet) else "net12.base."


if __name__ == "__main__":
    W = test_windows()
    for loader, name in ((load_gamma11, "gamma11_original_com"), (load_gamma15, "gamma15_fixed_com")):
        d = torch.load(MODELS / f"{name}.pt", weights_only=False)
        net, _ = loader()
        export(net, d["mu"], d["sd"], MODELS / f"{name}.mat", W)
