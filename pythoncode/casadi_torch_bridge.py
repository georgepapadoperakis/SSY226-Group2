"""Förstaderivator genom COM: CasADi beräknar, PyTorch tränar MLP.

Använder en sparad F(x,u,p) från MATLAB. Gamma är de sista 11 elementen i p.
Det motsvarar parameterordningen i er uppladdade create_dynamics_nmpcNew.m.
CPU/float64 används. Högre ordningens derivator stöds inte av denna brygga.
"""
import numpy as np
import casadi as ca
import torch
from torch import nn
from torch.autograd.function import once_differentiable


class _COMStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u, fixed, gamma, engine):
        # NumPy bryter normalt PyTorchs graf. Här anger vi uttryckligen
        # derivatorna i backward(), så gradientkedjan kan fortsätta genom COM.
        if any(t.device.type != "cpu" or t.dtype != torch.float64
               for t in (x, u, fixed, gamma)):
            raise ValueError("COM-bryggan kräver CPU och torch.float64")
        if u.requires_grad or fixed.requires_grad:
            raise ValueError("u och fixed behandlas som givna data, inte träningsvariabler")
        batch = x.shape[0]
        expected = ((batch, 9), (batch, 6), (batch, engine.nfixed), (batch, 11))
        if tuple(t.shape for t in (x, u, fixed, gamma)) != expected:
            raise ValueError(f"Fel COM-dimensioner; väntade {expected}")
        values, jx, jg = [], [], []
        arrays = [t.detach().numpy() for t in (x, u, fixed, gamma)]
        for i in range(batch):
            out = engine.evaluate(*(a[i] for a in arrays))
            values.append(np.asarray(out[0]).reshape(9))
            jx.append(np.asarray(out[1]))
            jg.append(np.asarray(out[2]))
        values, jx, jg = map(np.stack, (values, jx, jg))
        if not all(np.isfinite(a).all() for a in (values, jx, jg)):
            raise FloatingPointError("COM eller dess Jacobian gav NaN/Inf")
        ctx.save_for_backward(torch.from_numpy(jx), torch.from_numpy(jg))
        return torch.from_numpy(values)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        jx, jg = ctx.saved_tensors
        # Kedjeregeln: inkommande felgradient multipliceras med COM:s Jacobian.
        dx = torch.einsum("bi,bij->bj", grad_output, jx)
        dgamma = torch.einsum("bi,bij->bj", grad_output, jg)
        return dx, None, None, dgamma, None


class CasadiCOM(nn.Module):
    def __init__(self, function_file, sample_time=1.0):
        super().__init__()
        f = ca.Function.load(str(function_file))
        if f.n_in() != 3 or f.size1_in(0) != 9 or f.size1_in(1) != 6:
            raise ValueError("F måste ha indata x[9], u[6], p[n]")
        self.nfixed = f.size1_in(2) - 11
        if self.nfixed < 0 or sample_time <= 0:
            raise ValueError("Fel parameterstorlek eller samplingstid")
        x = ca.SX.sym("x", 9)
        u = ca.SX.sym("u", 6)
        fixed = ca.SX.sym("fixed", self.nfixed)
        gamma = ca.SX.sym("gamma", 11)
        p = ca.vertcat(fixed, gamma)
        dt = float(sample_time)
        # Samma RK4-princip som controller-koden: M=1, h=Ts.
        # est/ctrl är givna under respektive steg; gamma hålls under hela horisonten.
        k1 = f(x, u, p)[0]
        k2 = f(x + 0.5 * dt * k1, u, p)[0]
        k3 = f(x + 0.5 * dt * k2, u, p)[0]
        k4 = f(x + dt * k3, u, p)[0]
        next_x = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        self.evaluate = ca.Function(
            "mlp_com_step", [x, u, fixed, gamma],
            [next_x, ca.jacobian(next_x, x), ca.jacobian(next_x, gamma)],
        )

    def forward(self, x, u, fixed, gamma):
        return _COMStep.apply(x, u, fixed, gamma, self)


def rollout(engine, x0, u_sequence, fixed_sequence, gamma):
    """Fri simulering från x0. Gamma beräknas en gång före denna funktion.

    Här matas simulerat x vidare. Referenstillstånd återställs inte efter varje steg.
    """
    x = x0
    predictions = []
    for k in range(u_sequence.shape[1]):
        x = engine(x, u_sequence[:, k], fixed_sequence[:, k], gamma)
        predictions.append(x)
    return torch.stack(predictions, dim=1)
