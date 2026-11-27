The carrier plus relay passed the factual check; the failed marker version substantially changed two latent-input columns. That is a plausible failure source, not yet proof. I’ll preserve the safe mapping and recover the effect estimates and carrier construction. A useful projection must retain dose contrasts while staying quiet on held-out factual bundles.

```python
from pathlib import Path
import re
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

notes236 = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-BASED EVIDENCE")
print(notes236[:16000])

print("\nCURRENT EDITS")
safe236 = {}
for name in edited():
    a = load(name)
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    safe236[name] = np.asarray(a, dtype=np.float32).copy()
    b = load_original(name)
    if hasattr(b, "detach"):
        b = b.detach().cpu().numpy()
    print(name, safe236[name].shape,
          "change_norm", float(np.linalg.norm(safe236[name] - np.asarray(b))))

def describe236(x, depth=0):
    if isinstance(x, dict):
        return {
            str(k): describe236(v, depth + 1)
            for k, v in list(x.items())[:12]
        } if depth < 2 else list(x.keys())[:12]
    if isinstance(x, (tuple, list)):
        return [describe236(v, depth + 1) for v in x[:12]]
    if hasattr(x, "shape") and not isinstance(x, type):
        return {"type": type(x).__name__, "shape": tuple(x.shape)}
    return repr(x)[:140]

print("\nCARRIER AND RELAY OBJECTS")
for name in [
    "carrier211", "relay207", "rejected230", "orig208",
    "rp_X", "rp_input", "rr_input4", "cf215", "marker215",
    "B221", "B227", "P213", "pool213", "train213", "hold213"
]:
    if name in globals():
        print(name, describe236(globals()[name]))

print("\nIMPLEMENTATION HELPERS")
if "rx_show_function" in globals():
    for name in [
        "install_carrier200", "posterior211", "post208",
        "fixedpost215", "norm215"
    ]:
        try:
            rx_show_function(name, 4500)
        except Exception as exc:
            print(name, type(exc).__name__, str(exc)[:160])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## Run 236\n"
        "Preserved the factual-safe carrier211+relay weights in safe236. "
        "Recovering architecture, record-derived effect evidence, and carrier "
        "construction before deriving markers without replacing z_in columns. "
        "The large column differences are a suspected failure source, not an "
        "isolated causal test. No weights changed. "
        "Counts: 236 Python runs, 23 forwards.\n"
    )
```