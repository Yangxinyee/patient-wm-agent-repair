The validated glucose edit is restored exactly. The failed MAP addition was much larger than the retained regularized routes, although size alone cannot establish stability. I’ll test a half-strength, lower-norm MAP route. Passing factual NLL and improving both norepinephrine readouts would support keeping it.

```python
import numpy as np, torch, inspect

r281_base = np.asarray(load("core.b_in.weight")).astype(np.float32).copy()
r281_map = 0.5 * np.asarray(r272_trials[1.0]["correction"], dtype=np.float64)
r281_weights = (r281_base.astype(np.float64) + r281_map).astype(np.float32)
np.savez("/work/map_regularized_trial.npz",
         baseline=r281_base, map_correction=r281_map.astype(np.float32),
         core_b_in=r281_weights)
save_edit("core.b_in.weight",
          torch.from_numpy(r281_weights) if isinstance(load_original("core.b_in.weight"), torch.Tensor)
          else r281_weights)
r280_matrix_report("new regularized MAP addition", r281_map)
print("Current total edit norm",
      float(np.linalg.norm(np.asarray(load("core.b_in.weight")) -
                           np.asarray(load_original("core.b_in.weight")))))

for name in ["resume221_injection", "r267_full_pre"]:
    obj = globals().get(name)
    if callable(obj):
        print("\nHELPER", name)
        try:
            print(inspect.getsource(obj)[:2200])
        except Exception:
            print(inspect.signature(obj))

print("OTHER LOCAL HELPERS", [
    n for n in sorted(list(globals()))
    if callable(globals()[n]) and not n.startswith("_")
    and any(s in n.lower() for s in ["means", "prob", "evaluate", "prediction"])
][:35])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 281: lower-norm regularized MAP trial\n"
            "- Added half of retained ridge-1.0 minimum-Fisher MAP route to glucose strength 1.15.\n"
            "- Addition norm approximately 294.64 versus 3414.85 for the catastrophic trial.\n"
            "- Full-sequence factual and rollout validation required; no stability claim from norm alone.\n"
            "- Backup and trial saved in map_regularized_trial.npz.\n"
            "- Python runs: 281; forward passes: 26.\n")
```