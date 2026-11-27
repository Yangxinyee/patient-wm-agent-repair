The empty-action tests did not resolve the posterior mismatch, so that reconstruction is not yet reliable enough to support a weight edit. I’ll recover the architecture, earlier causal findings, and retained kernel objects before proceeding. This should reveal whether we can use the already-matched drug bundles without reconstructing every observation bundle.

```python
import json, inspect, numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== DATA README ===")
print(open("/data/README.md").read())

_notes28 = open("/work/notes.md").read()
_cut28 = _notes28.find("## Run 13")
print("=== EARLIER NOTEBOOK ===")
print(_notes28[:_cut28 if _cut28 >= 0 else 15000])

print("=== KERNEL INVENTORY (compact) ===")
for _name28, _obj28 in sorted(list(globals().items())):
    if _name28.startswith("_"):
        continue
    if isinstance(_obj28, (np.ndarray, torch.Tensor)):
        print(_name28, type(_obj28).__name__, tuple(_obj28.shape))
    elif isinstance(_obj28, pd.DataFrame):
        print(_name28, "DataFrame", _obj28.shape, list(_obj28.columns)[:18])
    elif isinstance(_obj28, dict) and len(_obj28) <= 250:
        print(_name28, "dict", len(_obj28), list(_obj28)[:14])
    elif isinstance(_obj28, (list, tuple)) and len(_obj28) > 5:
        print(_name28, type(_obj28).__name__, len(_obj28))
    elif inspect.isfunction(_obj28):
        try:
            print(_name28, "function", inspect.signature(_obj28))
        except Exception:
            print(_name28, "function")

with open("/work/notes.md", "a") as _f28:
    _f28.write(
        "\n## Run 28: recover architecture and causal evidence\n"
        "Empty-action alternatives did not verify the observation posterior. "
        "Recovering architecture, early causal estimates, and persistent objects "
        "before choosing a drug-bundle-specific edit. No weights edited.\n"
        "Python runs: 28; forward passes: 3.\n"
    )
```