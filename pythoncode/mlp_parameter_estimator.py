"""MLP för parameterfunktionen gamma(Tamb, x, Ib), med svenska kommentarer.

Featureordning: Tamb [degC], COM:s nio tillstånd, batteriström [A].
Utdataordning: p1,...,p11 i create_dynamics_nmpcNew.m.
Gränser och nominella parametrar måste hämtas från projektet före riktig träning.
"""
import torch
from torch import nn

STATE_NAMES = (
    "motT_K", "invT_K", "dcdcT_K", "batSoc", "batT_K",
    "compPin_Pa", "compPout_Pa", "cabIntT_K", "cabAirT_K",
)
FEATURE_NAMES = ("Tamb_C",) + STATE_NAMES + ("Ib_A",)
PARAMETER_NAMES = tuple(f"p{i}" for i in range(1, 12))


class ParameterMLP(nn.Module):
    def __init__(self, feature_mean, feature_std, lower, upper, nominal):
        super().__init__()
        # Buffertar sparas med modellen, men är inte träningsbara vikter.
        for name, value in (
            ("feature_mean", feature_mean), ("feature_std", feature_std),
            ("lower", lower), ("upper", upper),
        ):
            array = torch.as_tensor(value, dtype=torch.float64)
            if array.shape != (11,) or not torch.isfinite(array).all():
                raise ValueError(f"{name} måste innehålla 11 ändliga tal")
            self.register_buffer(name, array.clone())
        nominal = torch.as_tensor(nominal, dtype=torch.float64)
        if nominal.shape != (11,) or not torch.isfinite(nominal).all():
            raise ValueError("nominal måste innehålla 11 ändliga tal")
        if (self.feature_std <= 0).any() or (self.upper <= self.lower).any():
            raise ValueError("Standardavvikelser och parameterintervall måste vara positiva")
        if ((nominal <= self.lower) | (nominal >= self.upper)).any():
            raise ValueError("Nominella parametrar måste ligga strikt inom gränserna")

        # Linear beräknar W*z+b. Tanh tillför olinjäritet mellan lagren.
        # Det är en feedforward-arkitektur: nätverket har inget internt tidsminne.
        self.layers = nn.Sequential(
            nn.Linear(11, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 11),
        ).double()

        # Starta precis vid nominella gamma, oberoende av driftpunkt.
        # Första uppdateringen påverkar främst utlagret; därefter även tidigare lager.
        last = self.layers[-1]
        nn.init.zeros_(last.weight)
        fraction = (nominal - self.lower) / (self.upper - self.lower)
        with torch.no_grad():
            last.bias.copy_(torch.logit(fraction))

    def forward(self, features):
        if features.shape[-1] != 11:
            raise ValueError("MLP väntar 11 features i FEATURE_NAMES-ordningen")
        # Medelvärde/std ska beräknas på träningsdelen, inte validering eller test.
        z = (features - self.feature_mean) / self.feature_std
        raw = self.layers(z)
        # Sigmoid ger begränsade parametrar. Gränserna måste vara granskade;
        # begränsade gamma är inte i sig ett stabilitetsbevis för COM/NMPC.
        return self.lower + (self.upper - self.lower) * torch.sigmoid(raw)


def operating_point(x, tamb, ib):
    """Samla batchens driftpunkter; x har formen [batch, 9]."""
    return torch.cat((tamb.reshape(-1, 1), x, ib.reshape(-1, 1)), dim=-1)
