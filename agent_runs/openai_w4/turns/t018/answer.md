The architecture confirms that the value head uses a precise last-reading register, so the improved search result is consistent with a packing error. The output was truncated before the retained-variable details appeared. I’ll collect compact metadata and check state–reading alignment; this should distinguish an implementation mismatch from unavoidable latent-sampling differences.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== COMPACT RETAINED OBJECTS ===")
items = []
for key, obj in list(globals().items()):
    if key.startswith("_"):
        continue
    if isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray, torch.Tensor)):
        items.append(f"{key}:{type(obj).__name__}{tuple(obj.shape)}")
print("\n".join(items))

print("\n=== VOCAB METADATA ===")
for key in ("config", "codes", "event_kinds", "atom_kinds"):
    print(key, repr(VOCAB[key])[:1500])
num = VOCAB["numeric"]
print("numeric type:", type(num).__name__)
if isinstance(num, dict):
    for key, val in num.items():
        print("numeric", key, repr(val)[:850])
else:
    print(repr(num)[:3000])
print("value_bin_centres:", repr(VOCAB["value_bin_centres"])[:1800])

print("\n=== HEAD/VIEW TENSORS ===")
prefixes = (
    "core.gap", "core.out_norm", "core.flow", "core.z_prior",
    "core.z_head", "type_emb", "query_norm", "last_",
    "attr_heads.value_bin", "embed_transition"
)
for name in NAMES:
    if name.startswith(prefixes):
        print(name, SHAPES[name])

print("\n=== FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in sorted(folder.iterdir())])

state_path = Path("/work/fwd_4/states_original.npz")
state_meta = np.load(state_path)
print("\n=== STATE ARRAYS ===")
for key in state_meta.files:
    arr = state_meta[key]
    print(key, arr.shape, arr.dtype)
    if arr.ndim == 1:
        print(" first:", arr[:12])

fact_paths = sorted(Path("/work").glob("fwd_*/factual_original.csv"))
if fact_paths:
    fact_probe = pd.read_csv(fact_paths[0])
    print("\n=== FACTUAL COLUMNS AND EXAMPLE ===")
    print(list(fact_probe.columns))
    print(fact_probe.head(6).to_string(index=False))

print("\n=== TRAIN EXAMPLE ===")
first_stay = train["stay_id"].iloc[0]
print(train.loc[train.stay_id.eq(first_stay)].head(15).to_string(index=False))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 14\n"
        "Architecture explicitly specifies the four-field last_numeric register. "
        "Run 13 output was truncated; collected compact vocabulary, tensor and "
        "forward-file metadata to verify the CPU head and state alignment. "
        "No weights edited.\n"
    )
```