The architecture shows several normalization stages between the dose token and the value head, which may explain the weak signal. The recovery output was truncated before the useful helper details. I’ll inspect those interfaces and the recorded effect estimates next; this is inspection only, with no weight changes.

```python
import inspect, numpy as np
from pathlib import Path

notes182 = Path("/work/notes.md").read_text()
sections182 = notes182.split("\n## ")
print("NOTE LENGTH", len(notes182))
print("EARLY HEADINGS")
for s in sections182[:65]:
    print(s.splitlines()[0][:150])
print("\nACCEPTED EVIDENCE")
for s in sections182:
    heading = s.splitlines()[0].lower()
    if any(t in heading for t in ["run 19 ", "run 20 ", "run 21 ", "run 22 ", "run 23 ", "run 24 ", "run 25 ", "run 26 ", "run 27 ", "run 28 ", "run 29 ", "run 30 ", "run 31 ", "run 32 "]):
        print("\n##", s[:900])

print("\nWORKING FUNCTIONS")
for k in sorted(list(globals())):
    v = globals()[k]
    if callable(v) and getattr(v, "__module__", "") == "__main__" and (
        any(t in k.lower() for t in ["decoder", "feature", "step", "bundle", "cache", "factual", "wire", "head_input"])
    ):
        print(k, str(inspect.signature(v)), "names", v.__code__.co_names[:35])

print("\nSELECTED VARIABLES")
for k in ["kernels19", "kernel19", "chosen125", "QE125", "SP125", "Vnew125",
          "Wrepair125", "wire_out152", "routes145", "U167", "R118",
          "cf_features178", "cf_features180", "decoder_meta173", "fit179",
          "results179", "fact179", "factual178", "audit_ids", "audit_stays"]:
    if k not in globals():
        continue
    v = globals()[k]
    if isinstance(v, dict):
        print(k, {str(a): (tuple(b.shape) if hasattr(b, "shape") else str(b)[:160])
                  for a,b in list(v.items())[:15]})
    elif hasattr(v, "shape"):
        print(k, tuple(v.shape), "norm", float(np.linalg.norm(np.asarray(v))))
    else:
        print(k, str(v)[:500])

print("\nRELEVANT VARIABLE NAMES")
print([k for k in globals() if not k.startswith("_") and any(
    t in k.lower() for t in ["173", "174", "178", "179", "180", "kernels19", "factual", "audit", "quiet", "wire"]
) and not callable(globals()[k])][:180])

print("\nCURRENT WRITE SIZE",
      np.linalg.norm(B167-B167_original),
      np.linalg.norm(wire_out152-Wrepair125))
for name in ["decoder_capture180", "decoder_base180", "head_input180"]:
    if name in globals():
        f = globals()[name]
        try:
            print("\nSOURCE", name, inspect.getsource(f)[:3500])
        except Exception:
            print("\nFUNCTION DETAILS", name, f.__code__.co_varnames,
                  [str(x)[:700] for x in f.__code__.co_consts if isinstance(x, (str, tuple))])
```
