Earlier tests show that numeric amplification saturates, while several head and latent edits add almost no dose sensitivity. I’ll check whether the remaining error is mainly weak immediate response or insufficient persistence. Those require different edits; increasing persistence is only justified if the rollout curves support it.

```python
from pathlib import Path
import inspect, numpy as np, pandas as pd

print("TRANSITION AND REGISTER TENSORS")
for n in NAMES:
    if any(s in n for s in ["last_numeric", "value_bin", "query_norm"]):
        print(n, SHAPES[n])

print("\nROLLOUT FILE STRUCTURE")
roll91 = pd.read_csv("/work/fwd_10/dose_edit_edited.csv")
print(roll91.shape, roll91.columns.tolist())
print(roll91.head(5).to_string(index=False))

print("\nHOURLY PANEL")
print(panel_tuning.index.names, panel_tuning.columns.tolist())
print(panel_tuning.head(4).to_string())

print("\nVOCAB CODE MAP")
print(VOCAB["codes"])

print("\nDIRECTION HELPER")
fun91 = conditional_transition_slopes72
print(inspect.signature(fun91))
print("names:", fun91.__code__.co_names)
print("constants:", [x for x in fun91.__code__.co_consts
                     if isinstance(x, (str, int, float, tuple))][:12])

print("\nSAVED READOUT SURROGATE STRUCTURE")
for k91 in ["empirical_head59", "sensor_designs72", "repair_signal"]:
    obj91 = globals().get(k91)
    if isinstance(obj91, dict):
        print(k91)
        for key91, val91 in obj91.items():
            if isinstance(val91, dict):
                print(key91, {a91: (b91.shape if isinstance(b91, np.ndarray)
                                    else str(b91)[:70])
                               for a91, b91 in val91.items()})
            else:
                print(key91, val91.shape if isinstance(val91, np.ndarray)
                      else str(val91)[:80])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 91: response-versus-persistence audit\n"
            "- Previous numeric and latent gains saturated; direct-state/head edits did not materially improve rollouts.\n"
            "- Inspecting per-hour dose rollout files to estimate effective immediate response and accumulation separately.\n"
            "- No edits changed. Forward passes used: 34; 16 remain.\n")
```