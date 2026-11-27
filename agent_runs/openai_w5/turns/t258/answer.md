The prior work suggests latent sampling contributes to the discrepancy, but register timing or feature construction could also be wrong. I’ll inspect the reconstruction helpers and state alignment directly. A concrete convention error would justify fixing the diagnostics; otherwise calibration must rely on GPU tests.

```python
import inspect, dis, io
from pathlib import Path
import numpy as np

print("=== HEAD HELPER IMPLEMENTATIONS ===")
for name in ["make_head_inputs212", "head225"]:
    fn = globals().get(name)
    if fn is None:
        continue
    try:
        source = inspect.getsource(fn)
    except Exception:
        buf = io.StringIO()
        dis.dis(fn, file=buf)
        source = buf.getvalue()
    print(name, source[:3400])

print("\n=== STATE CONTAINERS ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, np.lib.npyio.NpzFile):
        print(name, [(k, obj[k].shape) for k in obj.files])
    elif isinstance(obj, dict) and name not in {"__builtins__", "W", "WO", "W225", "WO225", "BASE217", "V", "VOCAB", "SHAPES"}:
        print(name, "keys", list(obj.keys())[:10])

print("\n=== READING ALIGNMENT ===")
print("head_rows columns:", list(head_rows.columns))
print(head_rows.head(4).to_string(index=False))
print("GPU sample:")
print(gpu225.head(4).to_string(index=False))
for name in ["head_prev", "head_vb", "prev225", "rows225", "sf225", "s4", "s4e"]:
    if name in globals():
        obj = globals()[name]
        if isinstance(obj, np.ndarray):
            print(name, obj.shape, obj[:6])
        elif isinstance(obj, dict):
            print(name, list(obj.keys()))
        else:
            print(name, type(obj).__name__)

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 226–229: reconstruction audit\n"
            "- Re-read the architecture and record-identification notes; no weight edits.\n"
            "- Exact weight caches do not resolve local/GPU prediction disagreement.\n"
            "- Auditing prior sampling, register timing, and state alignment before using local likelihood calculations.\n")
```
