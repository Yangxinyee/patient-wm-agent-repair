The architecture confirms that drug actions reach the recurrent state through attention, whereas the earlier posterior edit barely reached it. No weights are edited. I’ll check the saved dose-extraction fits and record-derived evidence; a promising extractor should respond proportionally to changed doses but have little variation on factual inputs.

```python
from pathlib import Path
import numpy as np
import inspect

print("IDENTIFICATION NOTES")
print(Path("/work/notes.md").read_text()[:5200])

def summary100(v, depth=0):
    if isinstance(v, np.ndarray):
        a = v.astype(float)
        return {"shape": v.shape, "range": (float(a.min()), float(a.max())),
                "rms": float(np.sqrt(np.mean(a*a)))}
    if isinstance(v, dict):
        return {str(k): summary100(x, depth+1) for k, x in v.items()
                if depth < 2}
    if isinstance(v, (tuple, list)):
        return [summary100(x, depth+1) for x in v[:8]] if depth < 2 else f"length={len(v)}"
    return repr(v)[:180]

print("\nHIDDEN FIT STRUCTURE")
for k in [0.1, 1.0, 10.0, (1.0, 1e-8)]:
    if k in contrast_fits88:
        print(k, summary100(contrast_fits88[k]))
print("\nPOOLED HEAD-2 FITS")
for k in [(2, 0.1), (2, 1.0), (2, 10.0)]:
    if k in pool_fits89:
        print(k, summary100(pool_fits89[k]))
print("\nCHOSEN FITS", summary100(chosen90))
print("\nREADOUT SEARCH", best97)
print("First candidate:", readout_tests97[0])
print("Best candidates:", sorted(readout_tests97, key=lambda x: x[0])[:4])

print("\nHELPER CONSTANTS AND DEPENDENCIES")
for name in ["prelude_cache88", "pooled89", "state_update82", "memory82",
             "layer82", "action_vectors87", "linear", "layer_norm"]:
    f = globals()[name]
    print(name, "constants:", f.__code__.co_consts, "names:", f.__code__.co_names)

print("\nREADOUT TENSORS")
for n in NAMES:
    if any(s in n for s in ["attr_heads.value", "last_numeric", "last_status",
                           "query_norm", "type_emb", "z_head", "z_prior",
                           "embed_transition.value", "core.gap"]):
        print(n, SHAPES[n])
print("\nVOCAB TABLES", list(VOCAB.keys()))
print("CODES", VOCAB["codes"])
print("\nCURRENT EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\nPython runs 99-100; forwards 7. Original weights remain restored. "
            "Architecture re-read; recovering dose-extraction fits, identification "
            "evidence, and exact readout conventions before an effective recurrent edit.\n")
```
