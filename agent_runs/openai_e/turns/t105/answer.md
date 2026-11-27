The rejected NE edit changed factual predictions too much, and the posterior probes have not yet produced a selective correction. The insulin-only baseline remains installed. I’ll recover the architecture and retained helpers to choose a different NE edit. It must respond to NE dose changes while staying nearly inactive on factual and insulin-edited bundles.

```python
from pathlib import Path
import inspect, numpy as np, torch

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA SCHEMA")
print(Path("/data/README.md").read_text())

_notes90 = Path("/work/notes.md").read_text()
print("\nRELEVANT EARLIER NOTES")
for section in _notes90.split("\n## "):
    title = section.splitlines()[0]
    if any(f"Run {j}:" in title for j in range(47, 69)):
        print(("## " + section)[:1800])

print("\nCURRENT EDITS", edited())
print("\nEDIT / FORWARD HELPER INVENTORY")
for name, obj in sorted(list(globals().items())):
    if callable(obj) and hasattr(obj, "__code__"):
        if any(s in name.lower() for s in [
            "core", "coda", "head", "sens", "contrast", "action",
            "encode", "feature", "local", "update", "norm", "prelude", "fit"
        ]):
            print(name, inspect.signature(obj), "refs", obj.__code__.co_names)
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        if any(s in name.lower() for s in [
            "ne_", "81", "82", "83", "77", "delta", "feat",
            "design", "prelude", "contrast"
        ]):
            print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict) and any(s in name.lower() for s in [
        "81", "82", "83", "77", "candidate", "detector", "direction"
    ]):
        print(name, "dict keys", list(obj)[:25])

print("\nSAVED CANDIDATE ARRAYS")
for path in sorted(Path("/work").glob("*.npz")):
    with np.load(path, allow_pickle=False) as z:
        print(path.name, {k: z[k].shape for k in z.files})

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 90: restart recovery and selective-edit audit\n"
        "Insulin-only baseline77 remains installed. Recovering architecture, "
        "prior insulin validation, and retained NE/core helpers before a "
        "different NE correction. No weights changed. "
        "Python runs: 90; forwards: 12.\n"
    )
```