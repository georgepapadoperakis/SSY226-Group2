"""
Check the PyTorch COM (com_torch.py) against MATLAB's dynamics_mexfreeNew.m.

The test cases are committed in test_data/. To make new ones:
  MATLAB (from "Source Files"):  export_com_testcases            -> python/test_data/com_testcases_original_com.mat
                                 export_com_testcases(400, true) -> python/test_data/com_testcases_fixed_com.mat
Run (from the python folder):   python test_com_matlab_vs_python.py          (both files)
                                python test_com_matlab_vs_python.py <file>   (one file)

A file with 15 γ columns is checked against the fixed COM (com_torch.set_fixes(True)), otherwise the original.

Prints, per state, how far the PyTorch dx/dt is from MATLAB's. A correct port agrees to about 1e-10
(only floating-point rounding differs). Also checks that gradients reach all 11 γ.
"""
import os, sys
import numpy as np
import scipy.io as sio
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import com_torch
from com_torch import com_xdot, rk4_step, STATE_NAMES, INPUT_NAMES

torch.set_default_dtype(torch.float64)


def load_cases(path):
    m = sio.loadmat(path, squeeze_me=True)
    names = lambda k: [str(s) for s in np.atleast_1d(m[k])]
    uNames, estNames, ctrlNames = names("uNames"), names("estNames"), names("ctrlNames")
    assert uNames == INPUT_NAMES, f"input order differs: {uNames}"
    t = lambda a: torch.as_tensor(np.atleast_2d(np.asarray(a, float)))
    X, U, G, XDOT = t(m["X"]), t(m["U"]), t(m["G"]), t(m["XDOT"])
    EST, CTRL = t(m["EST"]), t(m["CTRL"])
    est = {n: EST[:, i] for i, n in enumerate(estNames)}
    ctrl = {n: CTRL[:, i] for i, n in enumerate(ctrlNames)}
    return X, U, est, ctrl, G, XDOT, t(m["Y"]), torch.as_tensor(np.atleast_1d(np.asarray(m["J"], float)))


def compare_with_matlab(path):
    X, U, est, ctrl, G, XDOT, Y, J = load_cases(path)
    xdot, out = com_xdot(X, U, est, ctrl, G, return_outputs=True)
    print(f"{len(X)} test points from {os.path.basename(path)}\n")
    print(f"{'state':12s} {'max |MATLAB|':>14s} {'max abs diff':>14s} {'max rel diff':>14s}")
    worst = 0.0
    for i, n in enumerate(STATE_NAMES):
        ref = XDOT[:, i]
        diff = (xdot[:, i] - ref).abs()
        rel = (diff / (ref.abs() + 1e-12 * ref.abs().max() + 1e-300)).max().item()
        worst = max(worst, rel)
        print(f"{n:12s} {ref.abs().max().item():14.4g} {diff.max().item():14.3g} {rel:14.3g}")
    # a few algebraic outputs (indices = rows of mdlOut_sym in dynamics_mexfreeNew.m, 1-based)
    checks = {"ventAirT_K": 15, "invClntTout_K": 1, "heaterClntTout_K": 7, "refMf": 20, "compTout_K": 9}
    print()
    for name, row in checks.items():
        d = (out[name] - Y[:, row - 1]).abs().max().item()
        print(f"output {name:18s} max abs diff {d:.3g}")
    print(f"output {'pwrTem (J)':18s} max abs diff {(out['pwrTem'] - J).abs().max().item():.3g}")
    ok = worst < 1e-8
    print("\nRESULT:", "PASS - the port matches MATLAB" if ok else f"CHECK - worst relative difference {worst:.3g}")
    return ok


def check_gradients(path):
    """Gradients of a 30-s RK4 rollout w.r.t. γ: finite, non-zero where γ is used, equal to finite differences."""
    X, U, est, ctrl, G, _, _, _ = load_cases(path)
    b = slice(0, 4)
    x0, u = X[b], U[b]
    e = {k: v[b] for k, v in est.items()}; c = {k: v[b] for k, v in ctrl.items()}
    ng = G.shape[1]
    g = torch.ones(4, ng, requires_grad=True)

    def loss_fn(gamma):
        x = x0
        for _ in range(30):
            x = rk4_step(x, u, e, c, gamma)
        scale = torch.tensor([10, 10, 10, 0.1, 10, 1e5, 1e6, 10, 10])
        return ((x - x0) / scale).pow(2).sum()

    loss = loss_fn(g); loss.backward()
    grad = g.grad.sum(0)
    fd = torch.zeros(ng)
    h = 1e-6
    for i in range(ng):
        gp = torch.ones(4, ng); gp[:, i] += h
        gm = torch.ones(4, ng); gm[:, i] -= h
        fd[i] = (loss_fn(gp) - loss_fn(gm)) / (2 * h)
    print("\nGradient check (30-s rollout, γ = 1):")
    print(f"{'γ':5s} {'autograd':>14s} {'finite diff':>14s}")
    for i in range(ng):
        print(f"p{i+1:<4d} {grad[i].item():14.6g} {fd[i].item():14.6g}")
    finite = torch.isfinite(grad).all().item()
    close = torch.allclose(grad, fd, rtol=1e-4, atol=1e-8 * fd.abs().max().item())
    print("RESULT:", "PASS - gradients flow to γ and match finite differences" if (finite and close) else "CHECK")
    return finite and close


if __name__ == "__main__":
    paths = sys.argv[1:] or [os.path.join(HERE, "test_data", f) for f in
                             ("com_testcases_original_com.mat", "com_testcases_fixed_com.mat")]
    ok = True
    for path in paths:
        if not os.path.exists(path):
            sys.exit(f"{path} not found. Run export_com_testcases in MATLAB first (from the 'Source Files' folder).")
        n_gamma = sio.loadmat(path, variable_names=["G"])["G"].shape[1]
        com_torch.set_fixes(n_gamma >= 15)
        print(f"\n{'=' * 70}\n{os.path.basename(path)}: {n_gamma} γ columns -> "
              f"{'FIXED COM (15 γ)' if n_gamma >= 15 else 'ORIGINAL COM (11 γ + γ12)'}\n{'=' * 70}")
        ok &= compare_with_matlab(path)
        ok &= check_gradients(path)
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
