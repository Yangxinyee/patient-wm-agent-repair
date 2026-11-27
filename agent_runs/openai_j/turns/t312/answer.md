The record estimates support opposite-signed insulin and norepinephrine effects, although the norepinephrine estimate needs stronger causal assumptions. The architecture may allow an extrapolation correction that leaves factual drug atoms unchanged. I’ll check the complete training-dose support and encoder features; insufficient separation would rule out that approach.

```python
from pathlib import Path
import re
import inspect
import numpy as np
import pandas as pd

print("ACTION ENCODER TENSORS")
for name in NAMES:
    if name.startswith("encode_action."):
        print(name, SHAPES[name])

support272 = {"insulin": set(), "norepinephrine": set()}
for chunk in pd.read_csv(
    "/data/train.csv", usecols=["event", "rate", "dose"], chunksize=250000
):
    for drug in support272:
        pairs = chunk.loc[chunk.event.eq(drug), ["rate", "dose"]].drop_duplicates()
        support272[drug].update(
            (float(r), float(d)) for r, d in pairs.itertuples(index=False, name=None)
        )

print("\nCOMPLETE TRAINING ACTION SUPPORT")
for drug, pairs in support272.items():
    pairs = sorted(pairs)
    print(drug, "unique rate/dose pairs:", len(pairs))
    if pairs:
        print("rate range:", min(p[0] for p in pairs), max(p[0] for p in pairs))
        print("dose range:", min(p[1] for p in pairs), max(p[1] for p in pairs))
        print("pairs:", pairs if len(pairs) <= 40 else pairs[:15] + pairs[-15:])

print("\nCACHED ACTION REPRESENTATION")
for key in ("repair_norms", "repair_codes", "cat_names", "atom_bins"):
    obj = globals().get(key)
    if isinstance(obj, dict):
        print(key, "keys:", list(obj)[:25])
    else:
        print(key, repr(obj)[:600])

if "repair_atoms" in globals():
    print("repair_atoms columns:", list(repair_atoms.columns))
    cols = [c for c in ("event", "is_end", "rate", "dose", "rate_unit", "dose_unit")
            if c in repair_atoms.columns]
    print(repair_atoms.loc[
        repair_atoms.event.isin(["insulin", "norepinephrine"]), cols
    ].drop_duplicates().head(12).to_string(index=False))

print("\nACTION-RELATED HELPER SIGNATURES")
helper_names272 = []
for key, obj in list(globals().items()):
    if inspect.isfunction(obj):
        refs = set(obj.__code__.co_names)
        if refs.intersection({
            "repair_set_encode", "repair_atom_vectors",
            "repair_changed_action", "repair_core_rows"
        }) or key in {
            "repair_set_encode", "repair_atom_vectors",
            "repair_changed_action", "repair_core_rows"
        }:
            helper_names272.append(key)
            params = [(p.name, p.kind.name)
                      for p in inspect.signature(obj).parameters.values()]
            print(key, params)
            print(" references:", ", ".join(obj.__code__.co_names)[:650])

text272 = Path("/work/notes.md").read_text()
sections272 = re.split(r"(?m)(?=^## )", text272)
print("\nPRIOR NUMERIC-GAIN RESULT")
for section in sections272:
    if "Doubling embed.numeric.weight" in section:
        print(section[:1100])
        break

print("\nRECENT ENCODER EXPERIMENTS")
matches272 = [
    s for s in sections272
    if any(k in s.lower() for k in
           ("posterior-only", "numeric gain", "action-path",
            "action path", "compensat", "embedding"))
    and "Run 271" not in s
]
for section in matches272[-5:]:
    print(section[:650])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 272: full action-support audit\n"
        "- Collected all training infusion rate/dose pairs and inspected "
        "action-encoder dimensions and cached reconstruction helpers.\n"
        "- Investigating a constrained statistical extrapolation correction; "
        "no weights changed.\n"
        "- Counts: 272 Python runs, 34 forward passes.\n"
    )
```