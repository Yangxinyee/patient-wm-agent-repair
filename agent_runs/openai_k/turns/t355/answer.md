The pooled shifts are −9.56/+5.46 for insulin and +0.68/−0.17 for norepinephrine. However, the new convolution calculation does not reproduce the earlier stored targets, so its interpretation needs auditing. I’ll recover the exact coefficients and averaging convention; reproducing the earlier targets would explain the discrepancy, otherwise those targets need revision.

```python
from pathlib import Path
import numpy as np, pandas as pd, re

print("=== EXACT RETAINED RUN-132 OBJECTS ===")
for _k in sorted(list(globals())):
    if "132" not in _k:
        continue
    _v = globals()[_k]
    print("NAME:", _k, "TYPE:", type(_v).__name__)
    if isinstance(_v, np.ndarray):
        print(" shape", _v.shape,
              "values", _v.tolist() if _v.size <= 64 else "(large array)")
    elif isinstance(_v, pd.DataFrame):
        print(" shape", _v.shape, "columns", _v.columns.tolist()[:20])
    elif callable(_v):
        _c = getattr(_v, "__code__", None)
        if _c is not None:
            print(" args", _c.co_varnames[:_c.co_argcount],
                  "references", _c.co_names, "constants", repr(_c.co_consts)[:500])
    else:
        print(repr(_v)[:1200])

print("=== POSSIBLE RETAINED EMPIRICAL KERNELS ===")
_printed = 0
for _k in sorted(list(globals())):
    if _k.startswith("_") or not any(s in _k.lower() for s in
            ["kernel", "theta", "psi", "beta", "coef", "response", "phys"]):
        continue
    _v = globals()[_k]
    if isinstance(_v, np.ndarray) and _v.size <= 48:
        print(_k, _v.shape, _v.tolist())
        _printed += 1
        if _printed >= 14:
            break

_notes = Path("/work/notes.md").read_text()
_sections = re.split(r"\n(?=## )", _notes)
print("=== EARLIER IDENTIFICATION / TARGET AUDIT HEADINGS ===")
_headers = []
for _s in _sections:
    _m = re.match(r"## Run (\d+)", _s)
    if _m and 12 < int(_m.group(1)) <= 134:
        _h = _s.splitlines()[0]
        if any(w in _h.lower() for w in
               ["record", "effect", "causal", "identif", "impulse",
                "dynamic", "convolution", "support", "physical", "target"]):
            _headers.append(_h)
print("\n".join(_headers)[:2000])

print("=== FINAL CANDIDATE FACTUAL MEANS ===")
_final_factual = pd.read_csv("/work/margin_paired400.csv")
print(_final_factual[["original_nll", "edited_nll", "delta_nll"]]
      .mean().to_string())
with np.load("/work/margin_candidate.npz", allow_pickle=False) as _a:
    for _k in ["glucose_correction", "map_correction"]:
        _sv = np.linalg.svd(_a[_k], compute_uv=False)
        print(_k, "norm", np.linalg.norm(_a[_k]),
              "largest singular values", _sv[:3])

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Run 296: audit discrepancy in empirical rollout targets\n"
             "- Pooled current rollouts: insulin x2/half -9.560353/+5.460528; "
             "NE x2/half +0.679243/-0.165227.\n"
             "- Run-6 six-lag convolution on actual reached readout rows yields "
             "insulin -12.720408/+6.360204 and NE +4.169603/-2.084802, "
             "but does not reproduce the older run-132 first-batch estimates.\n"
             "- Recovering exact run-132 coefficients and averaging convention before "
             "drawing a calibration conclusion; no weights changed.\n"
             "- Python runs: 296; forward passes: 44.\n")
```