The reconstruction error is larger than latent-sampling noise can explain, so I would not trust an edit derived from it yet. I’ll check the architecture, exported-state layout, and original rollout results. Matching the official predictions within sampling error would validate the reconstruction; otherwise, we need to correct its assumptions first.

```python
import os, json, numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== RECORD SCHEMA ===")
print(open("/data/README.md").read())

print("\n=== RETAINED FUNCTIONS ===")
for r26_name, r26_obj in list(globals().items()):
    if not r26_name.startswith("_") and callable(r26_obj):
        if getattr(r26_obj, "__module__", "") == "__main__":
            print(r26_name)

print("\n=== RETAINED DATA OBJECTS ===")
for r26_name, r26_obj in list(globals().items()):
    if r26_name.startswith("_") or r26_name.startswith("r26_"):
        continue
    if isinstance(r26_obj, (np.ndarray, pd.DataFrame, pd.Series, torch.Tensor)):
        print(r26_name, type(r26_obj).__name__, r26_obj.shape)
    elif isinstance(r26_obj, dict) and r26_name not in ("SHAPES", "VOCAB", "__builtins__"):
        print(r26_name, "dict", len(r26_obj), "keys:", list(r26_obj)[:18])

print("\n=== FORWARD FILES AND ROLLOUT SUMMARIES ===")
for r26_dir in ("/work/fwd_1", "/work/fwd_2", "/work/fwd_3"):
    print(r26_dir, os.listdir(r26_dir))
    for r26_file in os.listdir(r26_dir):
        r26_path = os.path.join(r26_dir, r26_file)
        if r26_file.endswith(".json"):
            print(r26_file, open(r26_path).read()[:6000])
        elif r26_file.startswith("dose_edit") and r26_file.endswith(".csv"):
            r26_df = pd.read_csv(r26_path)
            print(r26_file, "shape:", r26_df.shape, "columns:", r26_df.columns.tolist())
            print(r26_df.head(8).to_string(index=False))
        elif r26_file.startswith("states") and r26_file.endswith(".npz"):
            with np.load(r26_path) as r26_npz:
                print(r26_file, {k: (r26_npz[k].shape, str(r26_npz[k].dtype))
                                  for k in r26_npz.files})

with open("/work/notes.md", "a") as r26_note:
    r26_note.write(
        "\n## Run 26 / restart recovery\n"
        "No edits. CPU-head disagreement exceeds estimated latent MC noise, so an "
        "edit based on that reconstruction is not yet justified. Recovering exact "
        "architecture, retained objects, state schema, and original rollout results "
        "before choosing the correction pathway.\n"
    )
```