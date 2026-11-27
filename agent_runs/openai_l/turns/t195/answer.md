The accepted kernels show delayed, multi-hour effects, so matching only the next reading may not be enough. I’ll first collect decoder contrasts for stopping, halving, and larger doses, and check the existing record-based cross-drug estimates. This will test whether one selective readout can generalize across doses rather than fit doubling alone.
```python
import time
import numpy as np

print("JOINT RECORD-BASED EFFECT ESTIMATES:")
for n178 in ("effects14", "effects15"):
    if n178 in globals():
        print(n178, repr(globals()[n178])[:1800])

print("\nCOLLECTING CURRENT-EDIT CONTRASTS:")
start178 = time.time()
for drug178 in ("insulin", "norepinephrine"):
    for factor178 in (0.0, 0.5, 1.5, 3.0):
        make_cf81(drug178, factor178, batch_size=96)
        print(drug178, factor178, "elapsed_s", round(time.time() - start178, 1))

print("\nOWN-SENSOR CONTRASTS:")
for drug178, sensor178 in (("insulin", "glucose_finger"),
                          ("norepinephrine", "map_arterial")):
    base178 = cf_features81[(drug178, 1.0)]
    row178 = np.asarray(base178["rows"])
    mask178 = measure67.iloc[row178]["event"].to_numpy() == sensor178
    for factor178 in (0.0, 0.5, 1.5, 2.0, 3.0):
        case178 = cf_features81[(drug178, factor178)]
        shift178 = np.asarray(case178["means"]).mean(0) - np.asarray(base178["means"]).mean(0)
        rate178 = np.asarray(case178["delta_rate"])
        slope178 = np.dot(rate178[mask178], shift178[mask178]) / np.dot(rate178[mask178], rate178[mask178])
        print(drug178, factor178,
              "mean_shift", round(float(shift178[mask178].mean()), 5),
              "per_rate_slope", round(float(slope178), 5))

print("\nFACTUAL ALIGNMENT:")
print("observed bins", np.asarray(observed_bins82).shape,
      np.unique(np.asarray(observed_bins82)).tolist())
print("training stays", measure67.stay_id.nunique(),
      "first rows", measure67[["stay_id", "hour", "event"]].head(3).to_dict("records"))

with open("/work/notes.md", "a") as f178:
    f178.write("\n## Run 178 / generalized contrast features\n"
               "Collected current quiet-write decoder features for both drugs at x0, x0.5, "
               "x1.5, x2, and x3. Retained record kernels19; no new weights edited. "
               "Next fit a factual-anchored, code-selective final value readout.\n")
```