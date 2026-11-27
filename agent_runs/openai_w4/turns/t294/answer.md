The two independently fitted lag kernels agree closely, and their first-lag effects agree with the earlier estimates. That supports using them provisionally, but not treating them as known truth. I’ll verify their dose scale and the rollout file’s averaging convention. A mismatch in either would change the calibration targets.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("IDENTIFICATION DOSE FIXTURES")
print("pan:", type(pan).__name__, pan.shape, pan.index.names)
for drug255 in ("insulin", "norepinephrine"):
    entry255 = dml_residuals[drug255]
    print(drug255, "keys:", list(entry255.keys()))
    idx255 = np.asarray(entry255["indices"], dtype=np.int64)
    fixture255 = np.asarray(entry255["dose"], dtype=float)
    raw255 = pan[drug255].fillna(0).to_numpy()[idx255]
    print("n", len(idx255), "raw range", (raw255.min(), raw255.max()),
          "fixture range", (fixture255.min(), fixture255.max()),
          "maximum raw-rate mismatch", np.max(np.abs(raw255 - fixture255)))

print("\nLATEST FORWARD FILES")
latest255 = sorted(
    (p for p in Path("/work").glob("fwd_*") if p.is_dir()),
    key=lambda p: int(p.name.split("_")[-1]))[-5:]
for directory255 in latest255:
    print(directory255.name, [p.name for p in directory255.iterdir()])

dosefiles255 = sorted(
    Path("/work").glob("fwd_*/dose_edit_edited.csv"),
    key=lambda p: int(p.parent.name.split("_")[-1]))
latest_dose_path255 = dosefiles255[-1]
latest_dose255 = pd.read_csv(latest_dose_path255)
print("\nLATEST DOSE OUTPUT", str(latest_dose_path255))
print("shape", latest_dose255.shape, "columns", list(latest_dose255.columns))
print(latest_dose255.head(4).to_string(index=False))

print("\nRETAINED CALIBRATION HELPERS")
for name255 in ("dose_metrics192", "r243_reached", "r246_ids",
                "r250_passing", "r243_W", "r244_current"):
    value255 = globals().get(name255)
    if callable(value255):
        print(name255, "args:",
              value255.__code__.co_varnames[:value255.__code__.co_argcount],
              "names:", value255.__code__.co_names,
              "constants:", repr(value255.__code__.co_consts)[:650])
    elif isinstance(value255, dict):
        print(name255, "dict length", len(value255),
              "keys", list(value255.keys())[:12])
    elif isinstance(value255, pd.DataFrame):
        print(name255, value255.shape, list(value255.columns))
        print(value255.head(2).to_string(index=False))
    else:
        print(name255, repr(value255)[:450])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 255\n"
            "Recovered 12-lag raw-rate kernels: insulin approximately "
            "[-2.05,-2.48,-2.21,-1.80,-1.37,-1.01,-.75,-.58,-.42,-.29,-.25,-.08]; "
            "NE [3.14,2.56,1.49,.77,.47,.38,.23,-.008,.22,.087,.019,.064]. "
            "Checking exact dose fixtures and reached-hour output schema before "
            "matched calibration. No weights changed.\n")
```