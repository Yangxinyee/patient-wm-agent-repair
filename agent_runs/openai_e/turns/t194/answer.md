The insulin-only edit passes the sampled factual limit by just 0.00036 nats, so there is little room for a norepinephrine edit. Its excess loss is concentrated in glucose. I’ll inspect the local probability decoder and training metadata, then test a record-based calibration. Held-out improvement—not training improvement alone—will determine whether it is useful.

```python
import inspect, re, numpy as np
from pathlib import Path

funcnames165 = [
    k for k, v in list(globals().items())
    if callable(v) and getattr(v, "__module__", "") == "__main__"
]
print("FUNCTION NAMES:", ", ".join(funcnames165))

inspectnames165 = [
    k for k in funcnames165
    if re.search(r"(readout|metadata|reading|register|calib|prob)", k, re.I)
]
print("\nTARGET FUNCTIONS:", inspectnames165)
budget165 = 6200
for k in inspectnames165:
    fn165 = globals()[k]
    print("\nFUNCTION", k)
    try:
        src165 = inspect.getsource(fn165)
        print(src165[:min(2600, budget165)])
        budget165 -= min(len(src165), 2600, budget165)
    except Exception:
        print("parameters:", fn165.__code__.co_varnames[:fn165.__code__.co_argcount])
        print("globals:", fn165.__code__.co_names)
        print("constants:", repr(fn165.__code__.co_consts)[:900])
    if budget165 <= 0:
        break

print("\nTRAIN METADATA")
print(train_meta162.head(4).to_dict("records"))
print("STATE ARCHIVE:", s161.files)
for key in s161.files:
    a = s161[key]
    print(key, a.shape, a.dtype, repr(a[:2])[:150])
print("\nCALIBRATION RESULT SHAPES")
for j, a in enumerate(r160):
    print(j, type(a).__name__, getattr(a, "shape", None))
print("\nNLL BY EVENT")
print(pair159.groupby("event").mean(numeric_only=True).to_string()[:1500])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 165: probability decoder recovery\n"
        "Insulin-only NLL margin is .00036 nats on the first tuning sample. "
        "Inspected probability/readout helpers and training-state metadata for a "
        "record-based, algebraically installed calibration. No weights changed. "
        "Python runs: 165; forwards: 22.\n"
    )
```