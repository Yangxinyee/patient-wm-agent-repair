The earlier reconstruction checks found that infusion END atoms belong in the observation encoder. The newer helper appears to exclude all drug atoms, which could explain its error. I’ll verify that routing and the expected-dose fit layout; a substantial error reduction after retaining END atoms would confirm the suspected bug.

```python
import dis
import numpy as np
from pathlib import Path

print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text()[:2800])

print("\n=== OBSERVATION HELPER ROUTING ===")
dis.dis(rr58_observation_encoding, depth=0)

print("\n=== ATOM HELPER INTERFACE ===")
print("arguments:", r36_atom.__code__.co_varnames)
print("globals:", r36_atom.__code__.co_names)
print("constants:", repr(r36_atom.__code__.co_consts)[:1000])

print("\n=== SAMPLE BUNDLE ROWS ===")
print("measurement indices:", rr58_mi[:8])
for _rows62 in rr58_mrows[:3]:
    print(type(_rows62).__name__, repr(_rows62)[:650])
print("action presence totals:", rr48_present.sum(axis=0),
      "rates range:", rr48_rates.min(axis=0), rr48_rates.max(axis=0))
print("input/assignment agreement:",
      np.sqrt(np.mean((rr51_assign_X - rr48_input0[:, :384]) ** 2)))

print("\n=== EXPECTED-DOSE TUPLE LAYOUT ===")
for _drug62, _fit62 in rr51_assignment.items():
    print(_drug62, "tuple length", len(_fit62))
    for _j62, _part62 in enumerate(_fit62):
        if hasattr(_part62, "shape"):
            _arr62 = np.asarray(_part62)
            print(_j62, tuple(_arr62.shape),
                  "range", (float(_arr62.min()), float(_arr62.max())))
        else:
            print(_j62, repr(_part62)[:180])

print("\n=== INPUT / ENCODING ALIGNMENT OBJECTS ===")
for _nm62 in sorted(list(globals())):
    if _nm62.startswith(("r34", "r35", "r36", "r42", "r43", "r48")):
        _vv62 = globals()[_nm62]
        if hasattr(_vv62, "shape") and isinstance(_vv62, (np.ndarray, torch.Tensor)):
            print(_nm62, tuple(_vv62.shape))
        elif isinstance(_vv62, dict):
            print(_nm62, "dict", list(_vv62)[:4])
        elif isinstance(_vv62, list):
            print(_nm62, "list", len(_vv62))
        elif callable(_vv62):
            print(_nm62, "callable")
with open("/work/notes.md", "a") as _f62:
    _f62.write("\n## Run 62\n"
               "Investigating whether the newer observation helper wrongly drops infusion END atoms. "
               "Earlier latent reconstruction favored END-to-observation routing. No edits. "
               "Counts: 62 Python runs, 6 forward passes.\n")
```