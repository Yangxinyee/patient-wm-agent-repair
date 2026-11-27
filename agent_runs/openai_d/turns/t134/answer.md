Strength 2.6 increased NLL by 0.00655 on the fresh group, so it lacks a safe factual-fit margin. I’ll restore strength 2 and inspect how norepinephrine enters the model. The next hypothesis is that a dose-sensitive state direction can support a repair with little factual disturbance; weak or unstable sensitivity would argue against forcing one.

```python
import torch
import numpy as np
import inspect

for n117, a117 in r115_strength2.items():
    save_edit(n117, torch.from_numpy(a117.copy()))
print("Restored strength-2 insulin candidate.")

print("\n=== VOCABULARY STRUCTURE ===")
print("top-level keys", list(VOCAB))
for k117 in VOCAB:
    if any(s in k117.lower() for s in ["norm", "code", "unit"]):
        print(k117, repr(VOCAB[k117])[:1800])

print("\n=== EMBEDDING TENSORS ===")
for n117 in NAMES:
    if n117.startswith(("embed.", "encode_action.")):
        print(n117, SHAPES[n117])

print("\n=== ACTION RECORDS ===")
print(actions99.columns.tolist())
print(actions99[actions99.event == "norepinephrine"].head(4).to_string(index=False))

print("\n=== STATE AND HEAD RECOVERY ===")
print("states0 type", type(states0).__name__)
if hasattr(states0, "keys"):
    print("states0 keys", list(states0.keys()))
elif hasattr(states0, "files"):
    print("states0 files", states0.files)
for k117 in ["rh_eval", "rh_register", "norm96", "mlp96", "lin96"]:
    f117 = globals()[k117]
    print(k117, inspect.signature(f117),
          "dependencies", f117.__code__.co_names,
          "locals", f117.__code__.co_varnames)
print("context/register variables",
      sorted(k for k in globals()
             if not k.startswith("_") and
             any(w in k.lower() for w in
                 ["rh_", "rep_", "ctx", "prior", "register", "meta", "read_rows"])))

with open("/work/notes.md", "a") as f117_n:
    f117_n.write(
        "\n## Forward 14 / Run 117: strength 2.6 rejected provisionally\n"
        "On fresh tuning stays 2020000200..299, NLL is 1.62614 edited "
        "vs 1.61959 original, delta +0.00655. Restored strength-2 "
        "insulin fallback. Exploring controlled action-embedding state "
        "contrasts for a lower-factual-impact NE correction, rather than "
        "amplifying the weak observational NE innovation decoder.\n"
    )
```
