The original rollouts are much weaker than the record-based estimates: roughly −3.68 for doubled insulin and +0.47 for doubled norepinephrine. I’ll test whether the value head can detect doses that depart from the usual dosing policy. That could permit a selective correction; poor held-out decoding or substantial cross-sensor leakage would refute this approach.

```python
import inspect
import json
import numpy as np
from pathlib import Path

print("=== RECONSTRUCTION IMPLEMENTATION ===")
try:
    print(inspect.getsource(reconstruct_reading_head))
except Exception as exc:
    fn0 = reconstruct_reading_head
    print("Source unavailable:", type(exc).__name__)
    print("Globals:", fn0.__code__.co_names)
    print("Locals:", fn0.__code__.co_varnames)
    print("Constants:", [x for x in fn0.__code__.co_consts
                         if isinstance(x, (str, int, float, tuple))])

print("\n=== METADATA ===")
print(head_meta.columns.tolist())
print(head_meta.head(5).to_string(index=False))
print("state archive keys:", np.load("/work/fwd_3/states_original.npz").files)

print("\n=== VOCABULARY ===")
print("codes:", VOCAB["codes"])
print("config:", VOCAB["config"])
print("numeric:", json.dumps(VOCAB["numeric"], indent=2)[:4200])
print("event kinds:", VOCAB.get("event_kinds"))
print("atom kinds:", VOCAB.get("atom_kinds"))

print("\n=== SMALL-VARIABLE INVENTORY ===")
for n, o in list(globals().items()):
    if n.startswith("_"):
        continue
    if isinstance(o, (str, list, tuple)) and not callable(o):
        size0 = len(o)
        if size0 < 80:
            print(n, repr(o)[:350])
        else:
            print(n, type(o).__name__, "length", size0)

print("\n=== RESPONSE-KERNEL ESTIMATOR ===")
try:
    print(inspect.getsource(estimate_response_kernel))
except Exception:
    print("Globals:", estimate_response_kernel.__code__.co_names)
    print("Locals:", estimate_response_kernel.__code__.co_varnames)
    print("Constants:", [x for x in estimate_response_kernel.__code__.co_consts
                         if isinstance(x, (str, int, float, tuple))])
print("response_kernels:", {
    k: repr(v)[:400] for k, v in response_kernels.items()
})

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Runs 34–36: restart recovery\n"
            "- Re-read architecture and record schema. Original first-100 tuning "
            "rollouts: insulin x2 -3.680, half +2.018; NE x2 +0.465, half -0.871.\n"
            "- Proposed repair: value-head exponential tilt driven by decoded "
            "dose-policy innovations, with sensor gating and held-out checks.\n"
            "- Insulin long-horizon causal target remains uncertain. No edits; "
            "5 forward passes used.\n")
```
