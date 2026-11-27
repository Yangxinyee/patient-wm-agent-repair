The stronger edit exceeded the factual-fit limit. The current weaker edit has a smaller CPU penalty, but remains unvalidated on the GPU. I’ll recover the editing helpers and record-derived targets before testing it. Useful dose responses with a GPU NLL increase below 0.005 would support keeping it; exceeding that limit would rule it out.

```python
from pathlib import Path as _P296
import inspect as _I296
import re as _R296
import pandas as _PD296

print("ARCHITECTURE")
print(_P296("/model/ARCHITECTURE.md").read_text())

_g296 = list(globals().items())
print("\nEDIT_HELPERS")
for _n296, _v296 in _g296:
    if callable(_v296) and getattr(_v296, "__module__", "") == "__main__":
        if _R296.search(r"pair|inject|insulin|norepi|patch|screen|restore|commit|candidate|28[0-9]|29[0-9]", _n296, _R296.I):
            try:
                _s296 = str(_I296.signature(_v296))
            except Exception:
                _s296 = "?"
            print(_n296, _s296, "USES", getattr(getattr(_v296, "__code__", None), "co_names", ()))

print("\nRECENT_GLOBALS")
for _n296, _v296 in _g296:
    if _R296.search(r"backup|ids|28[0-9]|29[0-9]", _n296, _R296.I) and not callable(_v296):
        if isinstance(_v296, dict):
            print(_n296, "dict", list(_v296.keys())[:16])
        elif isinstance(_v296, (list, tuple)):
            print(_n296, type(_v296).__name__, len(_v296), repr(_v296)[:350])
        elif isinstance(_v296, (int, float, str)):
            print(_n296, repr(_v296)[:350])
        elif hasattr(_v296, "shape"):
            print(_n296, type(_v296).__name__, _v296.shape)

_notes296 = _P296("/work/notes.md").read_text()
print("\nRECORD_IDENTIFICATION_AND_TARGETS")
_hits296 = [
    _line296 for _line296 in _notes296.splitlines()
    if _R296.search(r"identif|causal|randomiz|instrument|confound|kernel|true effect|original.*rollout|baseline.*shift", _line296, _R296.I)
]
print("\n".join(_line296[:1500] for _line296 in _hits296[:10]))
print("\n".join(_line296[:1500] for _line296 in _hits296[-6:]))

print("\nLATEST_FORWARD_FILES")
for _d296 in sorted(_P296("/work").glob("fwd_*"), key=lambda _x296: int(_x296.name.split("_")[-1]))[-4:]:
    print(_d296.name, [x.name for x in _d296.iterdir()])
    for _f296 in _d296.glob("*.csv"):
        _df296 = _PD296.read_csv(_f296)
        print(_f296.name, _df296.shape, _df296.columns.tolist())
        if "stay_id" in _df296:
            print("STAYS", sorted(_df296.stay_id.unique().tolist())[:100])
        print(_df296.head(2).to_string(index=False))

with _P296("/work/notes.md").open("a") as _f296:
    _f296.write("\n## Run 296\nRecovered architecture, current editing helpers, record-identification notes and latest GPU output schemas after restart. No weights changed. Current insulin 3/0, gate 27, earlier-NE gain 0.5 candidate awaits GPU validation.\n")
```
