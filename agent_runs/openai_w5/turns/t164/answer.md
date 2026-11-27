The four-neuron approximation barely changed local likelihood, but the noisy dose proxies and variable tilt slopes do not yet establish a reliable repair. I’ll recover the saved calibration objects, record-derived effects, and exact computation path before changing weights. Correct timing and stable dose sensitivity would support this approach; mismatched exposures or unstable responses would refute it.

```python
import inspect, pathlib, re, types
import numpy as np

print("=== ARCHITECTURE ===")
print(pathlib.Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(pathlib.Path("/data/README.md").read_text())

notes_text = pathlib.Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK: IDENTIFICATION AND EFFECTS ===")
print(notes_text[:16000])

print("=== RELEVANT PERSISTENT OBJECTS ===")
pattern = re.compile(
    r"141|140|139|head|tilt|causal|effect|target|dose|private|priv|capacity|proxy|"
    r"clean|layout|neuron|regress|fit|record|panel|kernel|transport|distill",
    re.I,
)
inventory = []
for name, obj in list(globals().items()):
    if name.startswith("_") or not pattern.search(name) or isinstance(obj, types.ModuleType):
        continue
    if callable(obj):
        try:
            desc = str(inspect.signature(obj))
        except Exception:
            desc = "(signature unavailable)"
    elif hasattr(obj, "shape"):
        desc = f"shape={tuple(obj.shape)}"
    elif isinstance(obj, dict):
        desc = f"keys={list(obj.keys())[:24]}"
    elif isinstance(obj, (list, tuple)):
        desc = f"length={len(obj)}; first={repr(obj[:3])[:180]}"
    else:
        desc = repr(obj)[:180]
    inventory.append(f"{name}: {type(obj).__name__} {desc}")
print("\n".join(inventory[:300]))
print("Relevant objects:", len(inventory))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Restart after run 141\n"
        "- Four-neuron head distillation has negligible local likelihood cost; no causal head edit was installed in run 141.\n"
        "- Dose-proxy error and heterogeneous local tilt sensitivity remain unresolved.\n"
        "- Recovering architecture, record-derived identification results, and persistent calibration objects before another edit.\n"
    )
```
