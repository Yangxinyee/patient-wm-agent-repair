The records support substantial dose effects, unlike the original model’s weak or wrong-sign responses. The earlier posterior edit also used a dose-minus-policy signal, so I’ll check why it failed and whether its decoder handles stopping and scaling doses. Accurate contrasts across all tested factors are necessary before reusing it with the new transport head.

```python
import numpy as np
import torch
import dis
from pathlib import Path

print("=== RECORD-DERIVED KERNELS ===")
for key, value in kernel_refs.items():
    print(key, np.round(np.asarray(value), 5).tolist())

print("\n=== PREVIOUS POSTERIOR PILOT ===")
notes151 = Path("/work/notes.md").read_text()
budget151 = 3900
for line in notes151.splitlines():
    if any(f"run {k}" in line or f"runs {k}" in line for k in range(51, 81)):
        text = line[:900]
        if budget151 - len(text) < 0:
            break
        print(text)
        budget151 -= len(text)

print("\n=== LINEAR / NONLINEAR ACTION DECODER ERRORS ===")
xx151 = np.c_[np.asarray(U103), np.ones(len(U103))]
for label, coef in (("linear104", ld104), ("linear105", ld105)):
    pred151 = xx151 @ np.asarray(coef)
    print(label, "RMSE", np.round(
        np.sqrt(np.mean((pred151 - np.asarray(YY103))**2, axis=0)), 6))
print("YY first rows", np.round(np.asarray(YY103)[[0, 1, 4213, 8426]], 4))
print("scenario order", scenarios103)
n151 = len(U103) // len(scenarios103)
for j, scenario in enumerate(scenarios103):
    section151 = slice(j*n151, (j+1)*n151)
    err151 = np.asarray(action_pred105)[section151] - np.asarray(YY103)[section151]
    print(scenario, "nonlinear RMSE", np.round(
        np.sqrt(np.mean(err151**2, axis=0)), 5))

print("\n=== KEY INPUTS AND POSTERIOR-PILOT ARRAYS ===")
for name, obj in list(globals().items()):
    if (name.endswith("81") or name.endswith("53") or name.endswith("54")
            or name.endswith("55") or name.endswith("56")):
        if isinstance(obj, np.ndarray) or torch.is_tensor(obj):
            print(name, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "keys", list(obj.keys())[:8])
print("\nwt145 implementation:")
print(dis.Bytecode(wt145).dis()[:1200])
print("\nReserved-head current projection norms:")
q151 = np.asarray(load("core.prelude.0.attn.q.weight"))
kv151 = np.asarray(load("core.prelude.0.attn.kv.weight"))
o151 = np.asarray(load("core.prelude.0.attn.out.weight"))
print("Q head3", np.linalg.norm(q151[144:192]),
      "K head3", np.linalg.norm(kv151[144:192]),
      "V head3", np.linalg.norm(kv151[336:384]),
      "out head3", np.linalg.norm(o151[:, 144:192]))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\nPython run 151; forwards 10. Reviewing prior gated-posterior "
        "pilot failure and checking action decoder across x0/.5/1/1.5/2/3 "
        "before combining it with the factual-neutral transport head. "
        "No weights changed.\n"
    )
```