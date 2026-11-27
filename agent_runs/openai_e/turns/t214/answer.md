The earlier experiments already found the latent pathway too weak for a useful repair, so I won’t repeat them. A more promising issue is that the norepinephrine detectors were fitted on original-model states, while the insulin edit changes those states. I’ll compare their behavior on paired training states and inspect how to rebuild the detector inputs. Larger factual leakage on repaired states would support this hypothesis.

```python
import re, numpy as np, torch
from pathlib import Path

print("EMBEDDING / BUNDLE HELPERS")
for k184 in sorted(list(globals())):
    o184 = globals()[k184]
    if callable(o184) and hasattr(o184, "__code__") and re.search(
            r"(embed|encode|atom|bundle|poster|context)", k184, re.I):
        c184 = o184.__code__
        print(k184, c184.co_varnames[:c184.co_argcount], c184.co_names[:16])

print("\nBUNDLE INPUT ARRAYS")
for k184 in ["core_e_all", "core_u_all", "ctx_all101", "cf_C", "cf_E", "cf_U",
             "cf_H0", "cf_H_variants", "cf_D", "cf_a_lists", "cf_e_lists",
             "cf_frame", "state_df", "train_records161", "train_meta162"]:
    if k184 not in globals():
        continue
    o184 = globals()[k184]
    if hasattr(o184, "shape"):
        print(k184, tuple(o184.shape))
        if hasattr(o184, "columns"):
            print(" columns:", list(o184.columns))
    elif isinstance(o184, (list, tuple)):
        sample184 = o184[0] if len(o184) else None
        print(k184, "length", len(o184), "first type", type(sample184).__name__,
              "first shape", getattr(sample184, "shape", None))
        if isinstance(sample184, dict):
            print(" first keys:", list(sample184.keys()))
        elif isinstance(sample184, (list, tuple)) and sample184:
            atom184 = sample184[0]
            print(" first item:", type(atom184).__name__,
                  list(atom184.keys()) if isinstance(atom184, dict) else getattr(atom184, "shape", None))

print("\nSMALL RECORD SAMPLE")
print(train_records161.head(4).to_string(index=False))
print("\nDATA DEFINITIONS")
print(Path("/data/README.md").read_text()[:1900])

print("\nSTATE ARCHIVE CHRONOLOGY")
for label184, a184 in [("original", train_original168), ("insulin baseline", train_edited168)]:
    print(label184,
          "first stay/hour", list(zip(a184["stay_id"][:8].tolist(),
                                      a184["hour"][:8].tolist())),
          "first z norm", float(np.linalg.norm(a184["x"][0, -32:])))
print("INITIAL/TIME TENSORS",
      [(n, SHAPES[n]) for n in NAMES
       if re.search(r"(init|^gap|core\.slots|core\.flow)", n)])

with Path("/work/notes.md").open("a") as f184:
    f184.write("\n## Run 184: baseline-state detector robustness hypothesis\n"
               "Earlier latent sensitivity was weak, so no posterior edit attempted. "
               "Inspecting exact action/world encoders to rebuild prelude detector features "
               "on paired original and insulin-baseline training states from forward22. "
               "Hypothesis: original-state fits leak on the repaired-state distribution. "
               "No weights changed. Python runs: 184; forwards: 22.\n")
```
