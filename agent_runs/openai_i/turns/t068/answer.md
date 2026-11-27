The posterior reconstruction now matches readings and infusion endings almost exactly. That removes an important implementation uncertainty, but does not yet establish a useful weight edit. I’ll recover the retained transition helpers and causal estimates, then test a dose-specific correction. It must strengthen the record-supported response without worsening factual fit.

```python
import inspect, json, numpy as np, pandas as pd, torch
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLY NOTEBOOK: IDENTIFICATION EVIDENCE ===")
notes = Path("/work/notes.md").read_text()
print(notes[:7000])

print("\n=== RETAINED FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, inspect.signature(obj))
        except (TypeError, ValueError):
            print(name)

print("\n=== RETAINED DATA / STATE OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, "columns:", list(obj.columns)[:35])
    elif isinstance(obj, (np.ndarray, torch.Tensor)) and obj.ndim >= 2:
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict) and any(
        word in name.lower() for word in
        ["state", "start", "poster", "record", "lag", "effect", "encode", "recon"]
    ):
        print(name, "dict keys:", list(obj.keys())[:30])

print("\n=== TRANSITION / POSTERIOR / OUTPUT TENSORS ===")
for name in NAMES:
    if any(word in name.lower() for word in
           ["z_post", "prelude", "transition", "readout", "value", "head", "recur"]):
        print(name, SHAPES[name])

print("\nCurrent edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 60; forward 6 — exact bundle encoding recovered\n"
        "- Observation and infusion-END posterior reconstructions match exposed means "
        "to about 2e-7 RMS; no weights edited.\n"
        "- Recovering transition helpers and original causal evidence before constructing "
        "a dose-specific, reversible correction.\n"
    )
```