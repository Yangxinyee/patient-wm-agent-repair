The early record analyses support an immediate insulin–glucose effect near −2 per rate unit and a norepinephrine–MAP effect near +3, with effects persisting across hours. These remain identification-based estimates, not observed counterfactual truth. I’ll inspect the existing candidate’s two correction channels, then calibrate them separately with official rollouts rather than rely on the inconsistent local readout.

```python
import dis
import numpy as np, torch
from pathlib import Path

print("=== CANDIDATE CONSTRUCTION ===")
dis.dis(build_candidate89)

print("\n=== CHANNEL AND TARGET ARRAYS ===")
for k in ["coef88", "target88", "J88", "d88", "resp88", "carrier74",
          "decoders74", "diagnostics74", "diagnostics90", "zinW88", "zinb88"]:
    if k not in globals():
        continue
    v = globals()[k]
    if isinstance(v, (np.ndarray, torch.Tensor)):
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        print(k, a.shape, "norm:", float(np.linalg.norm(a)),
              "first:", a.reshape(-1)[:8])
    elif isinstance(v, dict):
        print(k, {str(a): (tuple(b.shape) if hasattr(b, "shape") else str(b)[:180])
                  for a, b in list(v.items())[:8]})
    else:
        print(k, str(v)[:800])

print("\n=== CANDIDATE BASELINE / CHANNEL DIFFERENCES ===")
for name in candidates90[0.0]:
    base = torch.as_tensor(candidates90[0.0][name])
    unit = torch.as_tensor(candidates90[1.0][name])
    original = load_original(name)
    print(name, tuple(base.shape),
          "baseline-original norm:", float((base-original).norm()),
          "unit-baseline norm:", float((unit-base).norm()))
    if name == "z_in.weight":
        for j in (30, 31):
            print(" channel", j, "base norm:", float(base[:, j].norm()),
                  "unit norm:", float(unit[:, j].norm()),
                  "difference norm:", float((unit-base)[:, j].norm()))

print("\n=== CANDIDATE NOTES ===")
text137 = Path("/work/notes.md").read_text()
for run137 in (74, 88, 89, 90):
    marker137 = f"## Run {run137}"
    pos137 = text137.find(marker137)
    if pos137 >= 0:
        end137 = text137.find("\n## ", pos137 + 4)
        print(text137[pos137:end137 if end137 >= 0 else pos137+1800][:1800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 137: inspect separable correction channels\n"
            "- Record estimates support drug-specific effects and persistence; remain observationally identified.\n"
            "- Inspecting baseline compression and channel directions before official calibration.\n"
            "- No edits. Python runs: 137; forward passes: 6.\n")
```