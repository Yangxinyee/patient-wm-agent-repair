The current edit predicts insulin shifts of −12.555/+4.182 and norepinephrine shifts of +0.587/−1.282, with factual NLL up 0.00402. The record-based norepinephrine target is roughly +4.2 to +4.6/−2.1 to −2.3, so simply scaling the existing correction is unlikely to fit within the remaining allowance. I’ll verify how the multi-direction sensitivities map to weights, then fit a more selective correction.

```python
import re, dis, io, types
from pathlib import Path
import numpy as np
import torch

arch140 = Path("/model/ARCHITECTURE.md").read_text()
pos140 = arch140.find("b_in")
if pos140 < 0:
    pos140 = arch140.lower().find("coda")
print("WEIGHT INTERFACE")
print(arch140[max(0, pos140 - 900):pos140 + 1800])

print("\nEXACT CODA HELPER")
buf140 = io.StringIO()
dis.dis(coda_input134, file=buf140)
print(buf140.getvalue()[:2400])
print("defaults:", [
    (type(v).__name__, tuple(v.shape) if isinstance(v, (np.ndarray, torch.Tensor)) else str(v)[:60])
    for v in (coda_input134.__defaults__ or ())
])

print("\nFACTUAL FEATURES AND NULL-SPACE BASES")
count140 = 0
for name140, obj140 in sorted(list(globals().items())):
    if name140.startswith("_") or not isinstance(obj140, (np.ndarray, torch.Tensor)):
        continue
    shape140 = tuple(obj140.shape)
    if (
        (len(shape140) == 2 and shape140[-1] == 188 and shape140[0] != 5988)
        or (len(shape140) == 2 and shape140 in [(192, 188), (188, 192)])
        or name140 in ["E_fact95", "E_base95", "prebasis136", "basis136",
                        "E_fact103", "F103", "F115", "FF103"]
    ):
        print(name140, shape140)
        count140 += 1
        if count140 >= 24:
            break

print("\nREADOUT HELPERS")
for name140, obj140 in sorted(list(globals().items())):
    if isinstance(obj140, types.FunctionType) and re.search(r"(read|head|decode|coda)", name140, re.I):
        if re.search(r"(12[0-9]|13[0-6])$", name140):
            print(name140, obj140.__code__.co_varnames[:obj140.__code__.co_argcount],
                  "uses", obj140.__code__.co_names[:16])

print("\nDRUG-TO-READING COVERAGE")
hm140 = head_meta.copy()
src140 = hm140["state_row"].to_numpy(dtype=int)
cfrows140 = np.asarray(drug_rows57, dtype=int)
for event140 in ["map_arterial", "map_cuff", "glucose_finger"]:
    print(event140, "drug rows with direct next-reading query:",
          len(set(src140[hm140["event"].eq(event140)]) & set(cfrows140)))

with open("/work/notes.md", "a") as f140:
    f140.write(
        "\n## Run 140: output-constrained weight-interface audit\n"
        "Recovered current readouts insulin -12.555/+4.182, NE +.587/-1.282, "
        "factual NLL increase .00402. Insulin-only baseline increase .00278. "
        "Record-supported NE schedule target +4.2..4.6/-2.1..2.3. "
        "Checking exact coda sensitivity interface and admission-null feature basis "
        "before a multi-direction closed-form correction. No weights changed. "
        "Python runs: 140; forwards: 15.\n"
    )
```