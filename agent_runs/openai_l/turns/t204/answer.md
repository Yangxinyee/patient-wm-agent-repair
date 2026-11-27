The GPU audit gave a +0.00546 NLL increase, just outside the limit, so the full-strength readout is not acceptable. I’ll reduce it to 70% strength and retest. This may improve the dose answers less, but factual fit is a hard constraint.

```python
import numpy as np, torch
from pathlib import Path

scale186 = 0.70
candidate186 = scale186 * candidate185
for name186, value186 in [
    ("attr_heads.value_bin.2.weight",
     head_original185["attr_heads.value_bin.2.weight"] + candidate186[:-1].T),
    ("attr_heads.value_bin.2.bias",
     head_original185["attr_heads.value_bin.2.bias"] + candidate186[-1])
]:
    save_edit(name186, np.asarray(value186, dtype=np.float32))
    w63_cache[name186] = torch.as_tensor(value186, dtype=torch.float32)

delta186 = nll179(L179 + np.einsum("bnd,dk->bnk", X179, candidate186)) - original_nll179
print("SHRUNK LOCAL NLL",
      "train", float(delta186[fitmask179].mean()),
      "held-out", float(delta186[valmask179].mean()))

print("\nLOCAL INTERFACES")
for k186 in ["transition64", "make_cf81", "head_inputs96", "nll179"]:
    f186 = globals()[k186]
    print(k186, "variables", f186.__code__.co_varnames,
          "names", f186.__code__.co_names)
print("\nLABEL/EVENT ARRAYS")
for k186 in ["events179", "stays179", "B179", "C179", "reading_df73",
             "readings73", "const_train67", "trans_train67", "centres_train67"]:
    if k186 in globals():
        a186 = globals()[k186]
        print(k186, type(a186).__name__, getattr(a186, "shape", None),
              str(a186[:2])[:230])
print("LAST-REGISTER VARIABLES",
      [k for k in globals() if any(t in k.lower() for t in ["lastbin", "last_bin", "lastvalue", "last_value", "reading"])][:55])

with Path("/work/notes.md").open("a") as f186:
    f186.write(
        "\n## Forward 10 / full-strength readout fails factual limit\n"
        "100 tuning stays, 4751 readings: original NLL 1.63164, edited 1.63710 "
        "(+0.00546), so readout179[1] is rejected at full strength. "
        "Run 186 stages 0.70 times that correction, retaining the prior "
        "factual-neutral quiet wire, pending a new paired audit.\n"
    )
print("\nCURRENT EDITS", edited())
```
