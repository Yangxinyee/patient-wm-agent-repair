The held-out decoders recover only part of the dose change; stronger amplification also introduces substantial factual noise. That makes a head-only correction unsafe. I’ll review the exact prediction path and retained effect estimates before choosing another route; a useful route must carry dose changes reliably without disturbing factual predictions.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED FUNCTIONS ===")
r167_globals = list(globals().items())
for r167_name, r167_obj in r167_globals:
    if inspect.isfunction(r167_obj) and not r167_name.startswith("_"):
        try:
            r167_sig = str(inspect.signature(r167_obj))
        except Exception:
            r167_sig = "?"
        print(r167_name, r167_sig)

print("\n=== RETAINED VARIABLE NAMES ===")
print(" ".join(sorted(
    k for k, v in r167_globals
    if not k.startswith("_") and not inspect.ismodule(v)
    and not inspect.isfunction(v) and not inspect.isclass(v)
)))

print("\n=== EARLY IDENTIFICATION NOTES ===")
r167_notes = Path("/work/notes.md").read_text()
r167_sections = r167_notes.split("\n## ")
for r167_sec in r167_sections:
    if any(w in r167_sec.lower() for w in
           ["identification", "instrument", "randomiz", "lagged effect",
            "distributed lag", "true effect", "causal estimate"]):
        print(r167_sec[:2400])
        if "Run 90" in r167_sec:
            break

print("\n=== CURRENT EDITS ===")
for r167_name in edited():
    r167_delta = load(r167_name) - load_original(r167_name)
    print(r167_name, SHAPES[r167_name],
          "delta_norm", float(np.linalg.norm(np.asarray(r167_delta))))

with open("/work/notes.md", "a") as r167_f:
    r167_f.write(
        "\n## Run 167: reject noisy head-only dose decoding\n"
        "- Held-out register-independent decoders recover incomplete dose changes; "
        "amplification incurs substantial original-query noise.\n"
        "- Reviewing exact architecture and retained record-based effects before "
        "choosing a safer route. Saved edits unchanged.\n"
        "- Python runs: 167; forward passes: 10.\n"
    )
```