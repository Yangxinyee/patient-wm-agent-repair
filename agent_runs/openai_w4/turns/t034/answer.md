The supplied output is still run 27; it does not show the newer probe. No new conclusion is warranted. Please run this minimal, uniquely marked check so I can recover the remaining diagnostics without truncation and continue toward a testable edit.

```python
import json
from pathlib import Path

print("RECOVERY_PROBE_28")
for folder in ["fwd_1", "fwd_2"]:
    print(folder, json.loads(Path(f"/work/{folder}/summary.json").read_text()))

print("PANEL COLUMNS:", list(pan.columns))
print("DML RESIDUAL STRUCTURE:")
for key, value in dml_residuals.items():
    print(key, type(value).__name__,
          list(value.keys()) if isinstance(value, dict) else getattr(value, "shape", None))

print("PERSISTENCE:", persistence_repair)
print("KERNELS:", provisional_kernels)
print("HEAD CONFIG:", head_cfg_repair)

print("ACTION/LATENT TENSORS:")
for name in NAMES:
    if name.startswith(("encode_action.", "core.z_post.", "core.z_in.", "core.b_in.")):
        print(name, SHAPES[name])

print("STATE BANKS:")
for bank in [state_o, state_train_repair]:
    print({key: getattr(value, "shape", type(value).__name__) for key, value in bank.items()})

print("EDITS:", edited())
```