The diagnostic stopped because `kernel_refs` contains structured results rather than a single numeric array; no weights changed. I’ll rerun with type checks. This does not alter the hypothesis: the key test is whether the decoder tracks dose changes reliably before they enter the new transport channel.

```python
import numpy as np
import torch
import dis
from pathlib import Path

print("=== RECORD REFERENCE STRUCTURE ===")
for key, value in kernel_refs.items():
    if isinstance(value, (tuple, list)):
        print(key, [(type(v).__name__, getattr(v, "shape", None),
                     len(v) if hasattr(v, "__len__") else None) for v in value])
    elif isinstance(value, dict):
        print(key, "keys", list(value.keys())[:10])
    else:
        print(key, type(value).__name__, getattr(value, "shape", None))
print("retained kernel:", np.round(np.asarray(kernel), 5).tolist())

print("\n=== PREVIOUS POSTERIOR PILOT ===")
notes152 = Path("/work/notes.md").read_text()
budget152 = 3200
for line in notes152.splitlines():
    if any(f"run {k}" in line or f"runs {k}" in line for k in range(51, 81)):
        text = line[:800]
        if len(text) > budget152:
            break
        print(text)
        budget152 -= len(text)

print("\n=== LINEAR DECODER INPUT CHECK ===")
for input_name in ("U103", "NU103", "XX103"):
    arr152 = np.asarray(globals()[input_name])
    xx152 = np.c_[arr152, np.ones(len(arr152))] if arr152.shape[1] == 192 else arr152
    for label, coef in (("ld104", ld104), ("ld105", ld105)):
        if xx152.shape[1] == np.asarray(coef).shape[0]:
            pred152 = xx152 @ np.asarray(coef)
            print(input_name, label, "RMSE", np.round(
                np.sqrt(np.mean((pred152 - np.asarray(YY103))**2, axis=0)), 6))

print("\n=== NONLINEAR DECODER BY SCENARIO ===")
n152 = len(U103) // len(scenarios103)
for j, scenario in enumerate(scenarios103):
    s152 = slice(j*n152, (j+1)*n152)
    err152 = np.asarray(action_pred105)[s152] - np.asarray(YY103)[s152]
    print(scenario, np.round(np.sqrt(np.mean(err152**2, axis=0)), 5))

print("\n=== KEY INPUT AND PILOT ARRAYS ===")
for name, obj in list(globals().items()):
    if any(name.endswith(s) for s in ("81", "53", "54", "55", "56")):
        if isinstance(obj, np.ndarray) or torch.is_tensor(obj):
            print(name, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "keys", list(obj.keys())[:6])

print("\n=== ORIGINAL WEIGHT HELPER ===")
print(dis.Bytecode(wt145).dis()[:1000])
print("\n=== RESERVED HEAD NORMS ===")
q152 = np.asarray(load("core.prelude.0.attn.q.weight"))
kv152 = np.asarray(load("core.prelude.0.attn.kv.weight"))
o152 = np.asarray(load("core.prelude.0.attn.out.weight"))
print("Q", float(np.linalg.norm(q152[144:192])),
      "K", float(np.linalg.norm(kv152[144:192])),
      "V", float(np.linalg.norm(kv152[336:384])),
      "out", float(np.linalg.norm(o152[:,144:192])))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\nPython runs 151-152; forwards 10. Run 151 diagnostic stopped "
        "on structured reference data; no weight changes. Recovered "
        "reference structure, prior posterior-pilot notes, decoder "
        "scenario checks, and reserved-head projection norms.\n"
    )
```