The architecture confirms that actions enter both the latent posterior and the recurrent prelude. Earlier simple amplifications failed, so I’ll look for a targeted dose-residual correction rather than amplify everything. First I’ll recover the available reconstruction helpers and state data; a useful correction must respond to dose changes without materially disturbing factual predictions.

```python
import re

print("=== NOTEBOOK SECTION SUMMARIES ===")
sections = re.split(r"(?m)^## ", Path("/work/notes.md").read_text())
for section in sections:
    lines = section.strip().splitlines()
    if lines:
        print("\n", lines[0])
        summary = "\n".join(lines[1:])[:420]
        print(summary)

print("\n=== FUNCTIONS ===")
for key in sorted(list(globals())):
    obj = globals()[key]
    if not key.startswith("_") and callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(key)

print("\n=== ARRAYS, TABLES, DICTIONARIES ===")
for key in sorted(list(globals())):
    if key.startswith("_"):
        continue
    obj = globals()[key]
    if isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(key, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict) and key not in {"SHAPES", "VOCAB", "__builtins__"}:
        print(key, "dict", list(obj.keys())[:8])

print("\n=== RELEVANT WEIGHTS ===")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_in", "prelude", "encode_action", "attr_heads.value", "core.b_in")):
        print(name, SHAPES[name])

print("\n=== STATE FILE SCHEMAS ===")
for k in range(1, 13):
    for path in Path(f"/work/fwd_{k}").glob("states*.npz"):
        with np.load(path) as f:
            print(str(path), {key: f[key].shape for key in f.files})
```
