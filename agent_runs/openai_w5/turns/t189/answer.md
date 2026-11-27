The corrected attention calculation shows nearly linear dose transport, so immediate action encoding is unlikely to explain the asymmetry. I’ll recover the calibration and state-decoding details next. If dose signals remain proportional at actual reading times, centering or recurrent feedback becomes the stronger explanation.

```python
from pathlib import Path
import inspect, types, re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== EARLIER RECORD-BASED FINDINGS ===")
_notes = Path("/work/notes.md").read_text()
print(_notes[:14000])

print("=== PERSISTENT HELPERS ===")
for _n, _v in sorted(list(globals().items())):
    if inspect.isfunction(_v) and getattr(_v, "__module__", "") == "__main__":
        try:
            print(_n, str(inspect.signature(_v)))
        except Exception:
            print(_n)

print("=== RELEVANT PERSISTENT OBJECTS ===")
_pat = re.compile(r"pair|dose|rate|sensor|tilt|head|feature|state|tune|hold|layout|cand|base|gain|clean|effect|target|transport|calib", re.I)
for _n, _v in sorted(list(globals().items())):
    if _n.startswith("_") or not _pat.search(_n) or callable(_v) or isinstance(_v, types.ModuleType):
        continue
    if hasattr(_v, "shape"):
        print(_n, type(_v).__name__, tuple(_v.shape))
    elif isinstance(_v, dict):
        print(_n, "dict keys:", list(_v)[:35])
    elif isinstance(_v, (list, tuple)):
        print(_n, type(_v).__name__, "len", len(_v), "sample", str(_v[:8])[:350])
    elif isinstance(_v, (str, int, float, bool)):
        print(_n, repr(_v)[:350])
```
