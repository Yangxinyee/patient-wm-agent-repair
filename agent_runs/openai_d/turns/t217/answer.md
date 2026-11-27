The new cohort fails the factual threshold: NLL increased by 0.00783, with losses in both glucose and arterial MAP. The trial is therefore not safe to submit yet. I’ll inspect its correction features and the persistence estimates before revising it. A safer edit must preserve the dose responses while reducing factual loss across cohorts, not just on the original validation set.

```python
from pathlib import Path
import numpy as np, inspect

text186 = Path("/work/notes.md").read_text()
for heading, limit in (("## Run 6", 3600), ("## Run 177", 1600)):
    pos = text186.find(heading)
    print("\n", heading, "\n", text186[pos:pos+limit] if pos >= 0 else "not found")

def np186(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

print("\nEDITED ROWS")
for name in head185:
    d = np186(load(name)) - np186(load_original(name))
    if d.ndim == 2:
        axis = 1 if name.endswith(".0.weight") else 0
        scores = np.max(np.abs(d), axis=axis)
        ids = np.flatnonzero(scores > 1e-7)
        print(name, d.shape, "changed", ids.tolist(), "max", float(np.max(np.abs(d))))
    else:
        print(name, d.shape, "changed", np.flatnonzero(np.abs(d)>1e-7).tolist(),
              "max", float(np.max(np.abs(d))))

print("\nNE CONSTRUCTION OBJECTS")
for k in ("r177_X", "r177_fit", "r177_val", "r177_probe_direction",
          "r177_beta", "r177_v", "r177_b", "r177_offset",
          "r177_gate_strength", "r177_target_multiplier", "r177_results",
          "r170_ne_patch", "c180_functions"):
    if k in globals():
        x = globals()[k]
        if isinstance(x, dict):
            print(k, {kk: (tuple(vv.shape) if hasattr(vv, "shape") else repr(vv)[:120])
                      for kk, vv in list(x.items())[:20]})
        elif hasattr(x, "shape") and np186(x).size > 20:
            print(k, type(x).__name__, x.shape)
        else:
            print(k, repr(x)[:1800])

print("\nCUSTOM FUNCTIONS")
lines186 = []
for k, v in list(globals().items()):
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        try:
            lines186.append(k + str(inspect.signature(v)))
        except Exception:
            pass
print("\n".join(lines186)[:2300])
print("\nEARLY NON-RUN VARIABLES\n", [
    k for k in list(globals())
    if not k.startswith(("_", "r", "c")) and k not in ("np", "pd", "torch")
][:100])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 186: stronger trial fails second factual cohort\nForward 26 on tuning 100..199: original 1.62044, edited 1.62827, delta +0.00783. Stronger trial is not accepted. Its two tested cohorts pool to approximately +0.00529, leaving no safety margin. Preserved in c185_current; inspecting correction and causal persistence before revision.\n")
```