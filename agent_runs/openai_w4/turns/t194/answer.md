The edit now produces substantial insulin responses (**−8.99** for doubling, **+11.09** for halving), but norepinephrine remains almost unchanged. Simply increasing the scale could worsen the insulin asymmetry. I’ll inspect the retained dose-response estimates and channel construction. Near-linear effects in the records paired with asymmetric channel encoding would implicate the encoding; record-supported nonlinearity would instead support keeping some asymmetry.

```python
import re, inspect, pathlib, numpy as np, pandas as pd

notes169 = pathlib.Path("/work/notes.md").read_text()
print("EARLY NOTES")
print(notes169[:5500])

print("\nRELEVANT RETAINED NAMES")
names169 = sorted(
    n for n in list(globals())
    if re.search(r"(137|136|kernel|target|channel|capture|connect|outer|quad|slope|replay|step)", n, re.I)
)
print(" ".join(names169)[:1800])

print("\nLOCAL-PROJECTION STRUCTURES")
for drug169 in ["insulin", "norepinephrine"]:
    item169 = lp_new136[drug169][1]
    print(drug169, type(item169).__name__)
    if isinstance(item169, dict):
        for key169, val169 in item169.items():
            if np.isscalar(val169):
                print(" ", key169, val169)
            else:
                arr169 = np.asarray(val169)
                print(" ", key169, "shape", arr169.shape,
                      "values", arr169.tolist() if arr169.size <= 16 else "(omitted)")

print("\nCHANNEL/REPLAY FUNCTION SIGNATURES")
for name169 in names169:
    fn169 = globals().get(name169)
    if inspect.isfunction(fn169) and re.search(r"channel|capture|replay|step|connect", name169, re.I):
        try:
            print(name169, inspect.signature(fn169))
        except Exception:
            pass

roll169 = pd.read_csv("/work/fwd_14/dose_edit_edited.csv")
print("\nROLLOUT FILE COLUMNS:", roll169.columns.tolist())
print("FIRST ROW:", roll169.iloc[0].to_dict())
print("OUTER166:", type(outer166).__name__, getattr(outer166, "shape", None))

with open("/work/notes.md", "a") as log169:
    log169.write(
        "\n\n## Run 169\n"
        "GPU pass 13, scale-2 edit: insulin x2 -8.992, half +11.09; "
        "NE x2 -0.099, half -0.468 (100 tuning stays). "
        "Original: -0.428, +0.096, -0.096, -0.488 respectively. "
        "Insulin free-rollout sensitivity improved, but asymmetry and absent NE repair "
        "make blind shared-scale increases unjustified. Recovering retained nonlinear "
        "identification and channel construction. No weights changed.\n"
    )
```
