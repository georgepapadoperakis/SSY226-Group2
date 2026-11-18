# Current data audit

Audited source: three uploaded numeric exports. No vehicle network has been trained.

| Ambient temperature | Native samples | Duration | Finite signals | Prepared known states |
|---|---:|---|---|---|
| -5 °C | 14964 | 0–3600 s | 19/19 | 5/9 |
| -7 °C | 14956 | 0–3600 s | 19/19 | 5/9 |
| -10 °C | 14770 | 0–3600 s | 19/19 | 5/9 |

Ts=1 s and N=30 are defined by the uploaded init_nmpc.m. Each prepared run has 3,601 time points.

The current exports have the same time vector for all signals within each run. Solver sampling is nonuniform; samples from distinct runs do not automatically align.

Missing states: SOC, compressor inlet pressure, compressor outlet pressure, cabin interior mass temperature.

Battery state candidate: time-aligned mean of four packs, matching the plotting code; verify the COM measurement mapping before training.

Missing additional data include battery current, six actual COM controls, stage-parameter construction, reviewed gamma bounds and the interpolation baseline.

The MLP has 1,803 trainable scalars. A synthetic multistep gradient check passed and synthetic rollout loss fell from 1.20418 to about 1.85e-5. These numbers validate software only.

Actual MATLAB COM export, its numerical equivalence and its gradients remain to be checked on the vehicle model.
