The records contain useful dose variation, but the simple state-space fits disagreed with the threshold evidence and were rejected. The later innovation estimates are the working evidence, not established truth. I’ll recover their checks and finish validating infusion-end encoding; disagreement across those checks would argue against amplifying the current drug response.

```python
import numpy as np
from pathlib import Path

print("=== IDENTIFICATION NOTES, RUNS 11–19 ===")
text48 = Path("/work/notes.md").read_text()
a48 = text48.find("## Run 11")
b48 = text48.find("## Run 20")
print(text48[a48:b48][:4800])

print("\n=== EXACT ACTION CONFIGURATION ===")
for name in ("best45", "layout44", "umap44", "defaults42", "catnames42",
             "code42", "empty43", "emap43", "counts42", "idx42",
             "atommeta42", "ctx42", "targetz42"):
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, {
            k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:160])
            for k, v in obj.items()
        })
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, list):
        print(name, "length", len(obj), "first", repr(obj[:3])[:450])
    else:
        print(name, repr(obj)[:650])

print("\n=== DRUG NORMALISATION ===")
print("VOCAB keys:", list(VOCAB))
for key in ("insulin", "norepinephrine"):
    try:
        print(key, repr(num42[key])[:900])
    except Exception as ex:
        print(key, type(ex).__name__)
print("num42 top-level keys:", list(num42)[:15])

print("\n=== STATE EXPORT SCHEMA ===")
for filename in ("/work/fwd_3/states_original.npz",
                 "/work/fwd_4/states_original.npz"):
    with np.load(filename) as st48:
        print(filename, {k: st48[k].shape for k in st48.files})
        for k in ("stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms"):
            if k in st48.files:
                print(k, st48[k][:8].tolist())

print("\n=== LATENT AND ACTION ENCODER SHAPES ===")
for name in NAMES:
    if name.startswith(("z_post", "z_prior", "encode_action", "z_in", "z_head")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 48\n"
        "Threshold and innovation evidence, not rejected simple SSM fits, "
        "remain the basis for causal targets. Recovering exact action metadata "
        "and all-bundle export schema to validate END atoms before fitting any "
        "factual-preserving edit. No edits.\n"
    )
```