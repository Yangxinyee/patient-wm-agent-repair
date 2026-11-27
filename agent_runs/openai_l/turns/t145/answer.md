The recovered notes support negative insulin and positive norepinephrine effects, while explicitly rejecting the conflicting state-space estimates. The long output hid the implementation details, so I’ll inspect those separately. I need the fitted correction to transmit those record-derived effects without disturbing ordinary predictions.

```python
import inspect, re, numpy as np, torch
print("=== FUNCTION INVENTORY ===")
fn_recovery = [(k, v) for k, v in list(globals().items())
               if inspect.isfunction(v) and not k.startswith("_")]
print(", ".join(k for k, v in fn_recovery))
for k, v in fn_recovery:
    if re.search(r"(11[5-9]|12[0-9]|130|131)$", k):
        print(k, inspect.signature(v))

print("\n=== COMPONENTS ===")
for k in ["P125", "T125", "QE125", "SP125", "Vnew125", "Wrepair125", "chosen125"]:
    if k not in globals():
        print(k, "MISSING")
        continue
    v = globals()[k]
    print(k, type(v).__name__,
          tuple(v.shape) if hasattr(v, "shape") else str(v)[:400])
    if isinstance(v, (np.ndarray, torch.Tensor)):
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        print("  norm", float(np.linalg.norm(a)), "range", float(a.min()), float(a.max()))
    elif isinstance(v, dict):
        print("  keys", list(v))

print("\n=== RECENT VARIABLE NAMES ===")
print(", ".join(k for k in globals() if re.search(r"(11[5-9]|12[0-9]|130|131)$", k)))

print("\n=== CAUSAL EVIDENCE / KERNEL SECTIONS ===")
nr = Path("/work/notes.md").read_text()
sections_recovery = re.split(r"(?=^## )", nr, flags=re.M)
matches_recovery = [s for s in sections_recovery
                    if any(w in s.lower() for w in ["eight-lag", "8-lag", "distributed lag",
                                                    "impulse", "kernel", "run 125", "run 130"])]
for s in matches_recovery[-8:]:
    print(s[:600])
print("\n=== LAST NOTEBOOK ENTRIES ===")
print(nr[-2200:])
```