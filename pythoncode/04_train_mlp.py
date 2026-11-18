"""Träningsmotor för KOMPLETTA, granskade COM-data. Se DATA_CONTRACT.md.

Körs först efter kompletterad export och godkänd COM-kontroll.
PARTIAL-filer från steg 02 accepteras inte och ersätts aldrig med nollor.
"""
from pathlib import Path
import argparse
import copy
import json
import numpy as np
import torch
from mlp_parameter_estimator import ParameterMLP, STATE_NAMES, operating_point
from casadi_torch_bridge import CasadiCOM, rollout

ROOT = Path(__file__).resolve().parents[1]


def load_complete(file, metadata):
    data = dict(np.load(file, allow_pickle=False))
    required = ("t_s", "x", "u", "p_fixed_horizon", "Ib_A", "T_amb_C", "gamma_interp",
                "state_names", "control_names", "fixed_parameter_names", "gamma_names",
                "reviewed_mapping")
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"Ofullständiga träningsdata {file}: saknar {sorted(missing)}")
    if not bool(data["reviewed_mapping"].item()):
        raise ValueError(f"Signal-/enhetsmappningen måste granskas: {file}")
    if data["T_amb_C"].size != 1:
        raise ValueError("Detta första upplägg väntar konstant utetemperatur per körning")
    t = data["t_s"]
    n = len(t)
    shapes = {"x": (n, 9), "u": (n-1, 6),
              "p_fixed_horizon": (n-1, int(metadata["prediction_horizon"]),
                                  len(metadata["fixed_parameter_names"])),
              "Ib_A": (n,), "gamma_interp": (n, 11)}
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Fel form eller NaN/Inf: {file}, {key}; väntade {shape}")
    if t.ndim != 1 or not np.isfinite(t).all() or not np.isclose(t[0], 0.0):
        raise ValueError("Tidsvektorn måste vara ändlig och börja vid 0 s")
    if not np.allclose(np.diff(t), metadata["sample_time_s"]):
        raise ValueError("Data och COM måste ha samma konstanta samplingstid")
    for key, expected in (("state_names", STATE_NAMES),
                          ("control_names", metadata["control_names"]),
                          ("fixed_parameter_names", metadata["fixed_parameter_names"]),
                          ("gamma_names", metadata["gamma_names"])):
        if list(data[key]) != list(expected):
            raise ValueError(f"Fel ordning: {file}, {key}")
    return data


def make_windows(runs, horizon, start_s, stop_s, stride):
    windows = []
    for run in runs:
        t = run["t_s"]
        for k in range(0, len(t)-horizon, stride):
            if t[k] >= start_s and t[k+horizon] <= stop_s:
                windows.append((run, k))
    if not windows:
        raise ValueError("Inga tränings-/valideringsfönster passar tidsintervallet")
    return windows


def batch(windows, horizon):
    # En batch innehåller flera oberoende korta simuleringar.
    def tensor(key, sliced=False, future=False):
        arrays = []
        for run, k in windows:
            arrays.append(run[key][k+1:k+horizon+1] if future else
                          run[key][k:k+horizon] if sliced else run[key][k])
        return torch.as_tensor(np.stack(arrays), dtype=torch.float64)
    x0 = tensor("x")
    tamb = torch.tensor([float(r["T_amb_C"].item()) for r, _ in windows], dtype=torch.float64)
    # Varje fönster har en egen parameterprognos konstruerad från driftpunkten
    # vid k och tillåtna störningsprognoser. Ta inte framtida referenstillstånd
    # och beräkna deras vätskeegenskaper: det skulle läcka måldata till COM.
    fixed = torch.as_tensor(np.stack([r["p_fixed_horizon"][k, :horizon] for r, k in windows]),
                            dtype=torch.float64)
    return (operating_point(x0, tamb, tensor("Ib_A")), x0, tensor("u", sliced=True),
            fixed, tensor("x", future=True), tensor("gamma_interp"))


def predict_loss(model, engine, windows, horizon, scale, baseline=False):
    features, x0, u, fixed, target, gamma_base = batch(windows, horizon)
    # Gamma beräknas EN gång per fönster och hålls under hela horisonten.
    gamma = gamma_base if baseline else model(features)
    predictions = rollout(engine, x0, u, fixed, gamma)
    state_error = ((predictions-target) / scale).square().mean()
    if not torch.isfinite(state_error):
        raise FloatingPointError("Prediktionsfelet är inte ändligt")
    return state_error, gamma, gamma_base


def evaluate(model, engine, windows, horizon, scale, baseline=False):
    # Medelvärde över alla batchar, viktat med deras verkliga antal fönster.
    total = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), 16):
            group = windows[start:start+16]
            loss, _, _ = predict_loss(model, engine, group, horizon, scale, baseline)
            total += loss.item() * len(group)
    return total / len(windows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "mlp_training_config.json")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Utvärdera -7 C EFTER att modellen valts på valideringsdata")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if not cfg.get("parameter_values_are_reviewed", False):
        raise ValueError("Ange granskade gammagränser, nominella värden och felvikter i config först")
    for key in ("gamma_lower", "gamma_upper", "gamma_nominal"):
        value = np.asarray(cfg.get(key), dtype=float)
        if value.shape != (11,) or not np.isfinite(value).all():
            raise ValueError(f"{key} måste innehålla 11 ändliga tal")
    scale = torch.tensor(cfg["state_error_scale"], dtype=torch.float64)
    if scale.shape != (9,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("state_error_scale måste innehålla 9 positiva tal i tillståndens enheter")
    horizon = int(cfg["horizon"])
    if horizon < 1 or horizon > 30:
        raise ValueError("Första upplägget stöder horisont 1..30")
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    base = ROOT / "Data" / "NN_training"
    metadata = json.loads((base / "COM" / "com_metadata.json").read_text(encoding="utf-8"))
    if float(metadata["sample_time_s"]) != 1 or int(metadata["rk4_substeps"]) != 1:
        raise ValueError("Träningsupplägget är granskat för Ts=1 s och en RK4-substep")
    runs = [load_complete(base / "complete" / f"mlp_run_T{i}.npz", metadata) for i in (5, 10)]
    if [float(r["T_amb_C"].item()) for r in runs] != [-5., -10.]:
        raise ValueError("Träningsfilerna ska motsvara -5 och -10 C")
    # Uppdelning FÖRE fönster: första 2700 s träning, efter 2730 s validering.
    # Ett mellanrum på 30 s separerar delarna. Detta är en preliminär tidsvalidering
    # inom samma körningar, inte ett test av en oberoende körcykel.
    train = make_windows(runs, horizon, 0, 2700, 15)
    val = make_windows(runs, horizon, 2730, 3600, 30)
    features = torch.cat([batch(train[i:i+16], horizon)[0] for i in range(0, len(train), 16)])
    model = ParameterMLP(features.mean(0), features.std(0, unbiased=False).clamp_min(1e-8),
                         cfg["gamma_lower"], cfg["gamma_upper"], cfg["gamma_nominal"])
    engine = CasadiCOM(base / "COM" / "vehicle_com.casadi", metadata["sample_time_s"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    baseline_val = evaluate(model, engine, val, horizon, scale, baseline=True)
    best, best_state, bad_epochs, history = float("inf"), None, 0, []
    generator = np.random.default_rng(cfg["seed"])
    for epoch in range(cfg["epochs"]):
        model.train()
        order = generator.permutation(len(train))
        total = 0.0
        for start in range(0, len(train), cfg["batch_size"]):
            group = [train[i] for i in order[start:start+cfg["batch_size"]]]
            optimizer.zero_grad()
            prediction_error, gamma, prior = predict_loss(model, engine, group, horizon, scale)
            # Svag prior mot interpolation för dåligt identifierade parametrar.
            # Detta är regularisering, inte gamma-etiketter som huvudmål.
            prior_penalty = ((gamma-prior)/(model.upper-model.lower)).square().mean()
            loss = prediction_error + cfg["prior_weight"] * prior_penalty
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Ogiltig MLP-gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item() * len(group)
        val_loss = evaluate(model, engine, val, horizon, scale)
        history.append({"epoch": epoch+1, "train_objective": total/len(train), "val_state_mse": val_loss})
        print(f"Epoch {epoch+1}: train={total/len(train):.6g}, val={val_loss:.6g}, interpolation={baseline_val:.6g}")
        if val_loss < best:
            best, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if bad_epochs >= cfg["patience"]:
            break
    if best_state is None:
        raise RuntimeError("Ingen giltig modell kunde väljas")
    model.load_state_dict(best_state)
    output = base / "trained_mlp"
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "config": cfg, "metadata": metadata}, output / "mlp_checkpoint.pt")
    result = {"validation_mlp_mse": best, "validation_interpolation_mse": baseline_val,
              "horizon": horizon, "history": history, "test_evaluated": False}
    if args.evaluate_test:
        test = load_complete(base / "complete" / "mlp_run_T7.npz", metadata)
        if float(test["T_amb_C"].item()) != -7.:
            raise ValueError("Testfilen ska motsvara -7 C")
        windows = make_windows([test], horizon, 0, 3600, 30)
        result.update(test_evaluated=True,
                      test_mlp_mse=evaluate(model, engine, windows, horizon, scale),
                      test_interpolation_mse=evaluate(model, engine, windows, horizon, scale, True))
    (output / "training_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved:", output)


if __name__ == "__main__":
    main()
