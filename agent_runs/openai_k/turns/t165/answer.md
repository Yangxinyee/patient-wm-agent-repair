The captured helper exposes all inputs needed for a finite-difference assay, and the candidate recipe preserves 30 latent coordinates while reserving two for drug innovations. My assumed readout feature layout was wrong, so I won’t route effects through it yet. I’ll check local reconstruction and identify the feature blocks first; near-exact matches would justify proceeding.

```python
import numpy as np
import torch

print("=== LOCAL RECONSTRUCTION ACCURACY ===")
out150, h150 = result148
print("out RMS:",
      float(np.sqrt(np.mean((out150.numpy()-sx[probe_indices148, :192])**2))))
print("h RMS:",
      float(np.sqrt(np.mean(
          (h150.numpy().reshape(-1,768)-sx[probe_indices148,192:960])**2))))

print("\n=== READOUT FEATURE LAYOUT ===")
for event150, m150 in models145.items():
    a150, b150, ids150 = m150["ao"], m150["base"], m150["ids"]
    print(event150,
          "base sample", np.round(b150[:2], 4).tolist(),
          "ao first", np.round(a150[:2,:15], 4).tolist(),
          "ao last", np.round(a150[:2,-15:], 4).tolist())
    v150 = V144o[m150["variant"]][ids150]
    denom150 = np.sum(v150*v150)
    matches150 = []
    for start150 in range(a150.shape[1]-191):
        block150 = a150[:,start150:start150+192]
        scale150 = float(np.sum(block150*v150)/denom150)
        error150 = float(np.sqrt(np.mean((block150-scale150*v150)**2)))
        matches150.append((error150, start150, scale150))
    print("best state blocks:", sorted(matches150)[:3])
    print("selected model coefficients/intercept:",
          m150["model"].coef_.shape, m150["model"].intercept_)

print("\n=== AUXILIARY READOUT DATA ===")
print("num145:", brief148(num145))
for name150 in [n for n in globals() if "143" in n]:
    obj150 = globals()[name150]
    if isinstance(obj150, (np.ndarray, torch.Tensor)):
        print(name150, tuple(obj150.shape))
    elif hasattr(obj150, "columns"):
        print(name150, obj150.shape, list(obj150.columns))
    elif isinstance(obj150, dict):
        print(name150, "keys", list(obj150)[:15])
    elif callable(obj150):
        print(name150, "callable")
for cf150 in CF89:
    print("drug", cf150["drug"], "dose contrast quantiles:",
          np.round(np.quantile(cf150["dc"], [0,.25,.5,.75,1]),4))

basew150 = fn93(0.0)
print("baseline weight keys:", list(basew150))
print("Baseline decoder innovation weight shape:", Binnov89.shape)
print("Latent channel factual scales:", np.std(innov_targets_audit, axis=0))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 150: local reconstruction and feature-layout checks\n"
            "- Exact captured update inputs are available; checking reconstruction "
            "against official original states.\n"
            "- Reserved drug innovation channels are retained; feature ordering must "
            "be established before computing new response directions.\n"
            "- No saved edits changed. Python runs: 150; forward passes: 10.\n")
```