We have paired head inputs and action-encoding caches, but no validated repair. I’ll recover the original rollout readouts and the latent-layer shapes now. If the original dose responses already match the record-based estimates, that would argue against a large correction.

```python
import os, pandas as pd
print("FORWARD ARTIFACTS:")
for k in range(1, 9):
    folder = f"/work/fwd_{k}"
    print(k, os.listdir(folder))
    for filename in os.listdir(folder):
        if filename.startswith("dose_edit_") and filename.endswith(".csv"):
            tab = pd.read_csv(folder + "/" + filename)
            print("DOSE FILE:", filename, "shape:", tab.shape, "columns:", tab.columns.tolist())
            print(tab.head(4).to_string(index=False))
            labelcols = [c for c in tab if any(s in c.lower() for s in ("edit", "intervention", "scenario"))]
            shiftcols = [c for c in tab if any(s in c.lower() for s in ("shift", "change", "delta"))]
            if labelcols and shiftcols:
                print(tab.groupby(labelcols)[shiftcols].mean().to_string())
print("\nINITIAL NOTES:")
print(open("/work/notes.md").read()[:2600])
print("\nLATENT TENSORS:")
for name in NAMES:
    if name.startswith("core.") and any(s in name for s in ("z_", ".z.", "posterior", "prior")):
        print(name, SHAPES[name])
print("\nACTION ENCODER SHAPES:")
for name in NAMES:
    if name.startswith("encode_action."):
        print(name, SHAPES[name])
print("\nHEAD HELPER OPERATIONS:", ce_make_heads.__code__.co_names)
print("PAIRED HEAD ARRAYS:", {k: getattr(v, "shape", None) for k, v in hp_original_head.items()})
print("REPLAY OBJECT NAMES:", [n for n in globals() if n.startswith("rx_") and ("step" in n or "test" in n or "replay" in n)])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 151\nFocused baseline-rollout and latent/action tensor recovery; broad inventories have been unhelpful because of output truncation. No edits. Counts: 151 Python runs, 8 forward passes.\n")
```