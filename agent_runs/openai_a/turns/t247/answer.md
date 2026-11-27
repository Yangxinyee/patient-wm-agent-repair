The ridge-1 trial failed the main test: doubling insulin shifts glucose by only −4.04, substantially weaker than the fallback’s −7.61. Its slightly better halving response does not offset that loss. I’ll restore the fallback and inspect the contrast construction. A useful next candidate must improve the actual doubling rollout, not merely its local CPU approximation.

```python
import inspect, dis
from pathlib import Path

for n, tensor in stronger6_checkpoint.items():
    save_edit(n, tensor.clone() if torch.is_tensor(tensor) else tensor.copy())
print("Restored stronger6_checkpoint.")

print("\nSELECTIVE-GRAFT GLOBAL DEPENDENCIES:")
fn = make_selective_contrast_graft
print(fn.__code__.co_names)
try:
    print(inspect.getsource(fn))
except (OSError, TypeError):
    dis.dis(fn)

print("\nR3 VARIABLES:")
for name, obj in sorted(list(globals().items())):
    if name.startswith("r3_"):
        if isinstance(obj, dict):
            print(name, "dict keys:", list(obj.keys())[:16])
        elif isinstance(obj, (int, float, str, bool, type(None))):
            print(name, repr(obj))
        else:
            print(name, type(obj).__name__, getattr(obj, "shape", None))

print("\nR3 CANDIDATE STRUCTURE:")
for key in ((0.3, 1.0), (1.0, 1.0)):
    obj = r3_candidates.get(key)
    print(key, type(obj).__name__)
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(" ", k, type(v).__name__, getattr(v, "shape", None),
                  repr(v)[:200] if isinstance(v, (float, int, str)) else "")
    elif isinstance(obj, (tuple, list)):
        print([(type(v).__name__, getattr(v, "shape", None)) for v in obj])

notes_now = Path("/work/notes.md").read_text()
sections = notes_now.split("\n## ")
selected = [s for s in sections
            if any(t in s.splitlines()[0].lower()
                   for t in ("contrast", "nullspace", "protected", "feedback"))]
print("\nCONTRAST/DYNAMICS NOTES:")
print("\n\n".join("## " + s for s in selected)[-6500:])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Ridge-1 rollout rejected; Python 215, forwards 23/50\n"
        "- First-100 paired rollout: insulin x2 -4.041, half +3.547; "
        "NE x2 -0.534, half -0.599.\n"
        "- Ridge-1 scale16 loses substantial main-readout accuracy relative "
        "to stronger6 (-7.613, +3.390). CPU local contrast was misleading.\n"
        "- Restored stronger6_checkpoint. Further trials require exact GPU "
        "rollout validation and factual-fit margin.\n"
    )
```
