The architecture confirms that dose edits affect shared recurrent state, so protecting only the value head is insufficient. The recovery output was truncated. I’ll retrieve the candidate’s exact geometry and cached stay IDs next; this will let us distinguish poor generalization from a feature-reconstruction error.

```python
import re, numpy as np
from pathlib import Path

print("=== ACCEPTED RECORD EVIDENCE ===")
for section in re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M):
    m = re.match(r"## .*?[Rr]un(?:s)? (\d+)", section)
    if m and 11 <= int(m.group(1)) <= 21:
        print(section[:650])

print("\n=== RECENT CANDIDATE OBJECTS ===")
lines223 = []
for name in sorted(list(globals())):
    m = re.search(r"(\d+)$", name)
    if not m or not 200 <= int(m.group(1)) <= 219:
        continue
    obj = globals()[name]
    if callable(obj):
        if hasattr(obj, "__code__"):
            lines223.append(f"{name}: function args={obj.__code__.co_varnames[:obj.__code__.co_argcount]} refs={obj.__code__.co_names}")
    elif isinstance(obj, np.ndarray):
        lines223.append(f"{name}: array {obj.shape}")
    elif isinstance(obj, dict):
        lines223.append(f"{name}: dict {list(obj)[:12]}")
    elif isinstance(obj, (float, int, str, list, tuple)):
        lines223.append(f"{name}: {str(obj)[:100]}")
print("\n".join(lines223)[:4300])

print("\n=== CACHE AND AUDIT STAY IDS ===")
for path in sorted(Path("/work").glob("fwd_*/states_*.npz")):
    with np.load(path) as d:
        ids = np.unique(d["stay_id"])
        print(str(path), "n=",len(ids), "first=",ids[:5].tolist(), "last=",ids[-5:].tolist())
for name in ["measure67", "audit188", "audit168"]:
    if name in globals():
        d = globals()[name]
        ids = np.unique(d["stay_id"])
        print(name, "n=",len(ids), "first=",ids[:8].tolist(), "last=",ids[-5:].tolist())
print("\n=== IMPORTANT INTERFACES ===")
for name in ["kernels19", "raw208", "cases208", "candidates218",
             "backup219", "w63", "train", "tuning", "P", "A"]:
    if name in globals():
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, "keys:", list(obj)[:25])
        else:
            print(name, type(obj).__name__, getattr(obj, "shape", None))
print("value-head tensor shapes:", {
    n:SHAPES[n] for n in NAMES if n.startswith("attr_heads.value_bin")
})
```