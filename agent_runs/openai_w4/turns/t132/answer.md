The channel-only edit leaves factual NLL essentially unchanged: 1.59518 versus 1.59524. I’ll now test glucose-specific state directions using the measured sensitivities and the record-derived immediate effect. A viable direction should move glucose in the right amount while producing little MAP change, including under finite—not just infinitesimal—perturbations.

```python
import torch, numpy as np, time
rW = {nm: ww.clone() for nm, ww in channel_base117.items()}
probe_events118 = probe_data95["event"].to_numpy()
target118 = np.where(
    np.isin(probe_events118, ["glucose_finger", "glucose_lab"]),
    causal_beta80["insulin"], 0.0
)
JJ118 = np.asarray(J110, dtype=np.float64)
directions118 = {}
for ridge118 in [0.0001, 0.001, 0.01, 0.1]:
    vv118 = np.linalg.solve(
        JJ118.T @ JJ118 + ridge118 * np.eye(192),
        JJ118.T @ target118
    )
    directions118[ridge118] = vv118
    pp118 = JJ118 @ vv118
    print("ridge", ridge118, "norm", round(float(np.linalg.norm(vv118)), 2),
          "linear RMSE", round(float(np.sqrt(np.mean((pp118-target118)**2))), 3),
          "event means", {
              ev: round(float(pp118[probe_events118 == ev].mean()), 3)
              for ev in np.unique(probe_events118)
          })

base_state118 = transition73(probe_indices95)
base_means118 = np.asarray(probe_means95(base_state118))
original_linear118 = rp_linear
finite_results118 = {}
for ridge118 in [0.001, 0.01]:
    vv118 = directions118[ridge118]
    for q118 in [-2.0, -0.5, 0.5, 2.0]:
        shift118 = torch.as_tensor(q118 * vv118, dtype=torch.float32)
        def shifted_linear118(x, prefix):
            yy = original_linear118(x, prefix)
            return yy + shift118 if prefix == "core.b_in" else yy
        try:
            rp_linear = shifted_linear118
            ss118 = transition73(probe_indices95)
        finally:
            rp_linear = original_linear118
        dd118 = np.asarray(probe_means95(ss118)) - base_means118
        finite_results118[(ridge118, q118)] = dd118
        print("finite", ridge118, q118, {
            ev: (round(float(dd118[probe_events118 == ev].mean()), 3),
                 round(float(dd118[probe_events118 == ev].std()), 3))
            for ev in np.unique(probe_events118)
        })
print("Current saved edits:", edited())
with open("/work/notes.md", "a") as nf:
    nf.write("\n## Forward 5 / Run 118\n"
             "Channel-only factual validation, first 100 tuning stays: "
             "original 1.59524, edited 1.59518 (delta -0.00006). "
             "Now testing B_in response directions against record-based "
             "insulin immediate coefficient -2.04924, with MAP target zero. "
             "Finite perturbations are design checks, not causal evidence.\n")
```
