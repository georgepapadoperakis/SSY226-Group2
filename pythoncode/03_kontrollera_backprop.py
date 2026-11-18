"""Körbar teknisk kontroll. Alla siffror här är SYNTETISKA, inte projektresultat.

Kontrollen använder en enkel testdynamik för att verifiera hela kedjan:
MLP -> gamma -> CasADi/RK4 -> flerstegsfel -> backward -> viktuppdatering.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import casadi as ca
import torch
from mlp_parameter_estimator import ParameterMLP
from casadi_torch_bridge import CasadiCOM, rollout


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    dtype = torch.float64
    with TemporaryDirectory() as folder:
        # Samma in-/utdimensioner som projektet, men ENDAST en leksaksdynamik.
        x = ca.SX.sym("x", 9)
        u = ca.SX.sym("u", 6)
        p = ca.SX.sym("p", 12)  # en given parameter + elva gamma
        g = p[1:]
        dx = -0.2*x + g[:9] + 0.05*g[9]*x + 0.03*g[10]*x*x
        f = ca.Function("synthetic_only", [x, u, p], [dx, x, ca.SX(0)])
        file = Path(folder) / "synthetic.casadi"
        f.save(str(file))
        engine = CasadiCOM(file, sample_time=1.0)
        x0 = torch.randn(4, 9, dtype=dtype) * 0.1
        controls = torch.zeros(4, 5, 6, dtype=dtype)
        fixed = torch.zeros(4, 5, 1, dtype=dtype)

        # Numerisk gradientkontroll över flera steg, för både tillstånd och gamma.
        check_x = x0[:1].clone().requires_grad_()
        check_g = torch.full((1, 11), 0.7, dtype=dtype, requires_grad=True)
        passed = torch.autograd.gradcheck(
            lambda a, b: rollout(engine, a, controls[:1], fixed[:1], b),
            (check_x, check_g), atol=1e-5, rtol=1e-3,
        )
        assert passed
        print("PASS: numerical gradient check through multi-step simulation")

        # Dessa gränser är testvärden. De är INTE valda för de verkliga p1-p11.
        model = ParameterMLP([0.]*11, [1.]*11, [0.1]*11, [2.]*11, [1.]*11)
        features = torch.randn(4, 11, dtype=dtype)
        truth_gamma = torch.linspace(0.5, 1.5, 11, dtype=dtype).repeat(4, 1)
        with torch.no_grad():
            target = rollout(engine, x0, controls, fixed, truth_gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        losses = []
        for _ in range(120):
            optimizer.zero_grad()
            gamma = model(features)
            prediction = rollout(engine, x0, controls, fixed, gamma)
            loss = ((prediction-target)**2).mean()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0] * 0.05
        print("Architecture: 11 -> 32 -> 32 -> 11")
        print("Trainable weights and biases:", sum(p.numel() for p in model.parameters()))
        print(f"Synthetic loss: {losses[0]:.6g} -> {losses[-1]:.6g}")
        print("PASS: backpropagation reaches MLP weights and reduces synthetic rollout error")
        print("This is a software test, not training or validation of the vehicle COM.")


if __name__ == "__main__":
    main()
