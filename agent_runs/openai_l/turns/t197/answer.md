The final-layer fit cannot reach the record-based effects within the factual-loss limit: stronger corrections quickly worsen held-out NLL. I’ll inspect the earlier decoder inputs to see whether the dose wire survives there but is obscured by the existing nonlinear value layer. A cleaner signal there would justify a selective first-layer edit; otherwise the wire needs redesigning.
```python
import numpy as np
from pathlib import Path

probe180 = head_inputs96(ctx_edited173[:2], const_train67[:2], epsread73[:, :2])
print("Head-input helper:", type(probe180).__name__, tuple(probe180.shape))
print("Agreement with saved factual input:",
      float(np.sqrt(np.mean((np.asarray(probe180) -
                             np.asarray(input_edited173[:, :2]))**2))))

decoder_base180 = decode67
capture180 = {}
def decoder_capture180(ctx, const, trans, centres, eps, return_features=False):
    capture180["input"] = head_inputs96(ctx, const, eps).clone()
    return decoder_base180(ctx, const, trans, centres, eps,
                           return_features=return_features)

decode67 = decoder_capture180
try:
    for drug180 in ("insulin", "norepinephrine"):
        for factor180 in (1.0, 0.0, 0.5, 1.5, 2.0, 3.0):
            make_cf81(drug180, factor180, batch_size=96)
            cf_features81[(drug180, factor180)]["input"] = capture180["input"]
finally:
    decode67 = decoder_base180

I180 = np.asarray(input_edited173, dtype=np.float64)
R180 = np.asarray(R118, dtype=np.float64)
print("\nPrivate-direction factual input SD:",
      np.std((I180 @ R180).reshape(-1, 4), axis=0).round(5).tolist())
print("Private-direction basis shape:", R180.shape)
print("\nDose-contrast projections per physical rate unit:")
for drug180, sensor180 in (("insulin", "glucose_finger"),
                          ("norepinephrine", "map_arterial")):
    base180 = cf_features81[(drug180, 1.0)]
    rr180 = np.asarray(base180["rows"], dtype=int)
    mm180 = measure67.iloc[rr180].event.to_numpy() == sensor180
    for factor180 in (0.0, 0.5, 1.5, 2.0, 3.0):
        case180 = cf_features81[(drug180, factor180)]
        diff180 = (np.asarray(case180["input"], dtype=np.float64)
                   - np.asarray(base180["input"], dtype=np.float64)).mean(0)
        dr180 = np.asarray(case180["delta_rate"], dtype=float)
        slopes180 = (dr180[mm180, None] * (diff180 @ R180)[mm180]).sum(0)
        slopes180 /= np.dot(dr180[mm180], dr180[mm180])
        print(drug180, factor180, slopes180.round(6).tolist())

print("\nPREVIOUS READOUT/WIRE FINDINGS:")
parts180 = Path("/work/notes.md").read_text().split("\n## ")
for p180 in parts180:
    if any(p180.startswith(s) for s in
           ("Run 83", "Run 84", "Run 96", "Run 97", "Run 103", "Run 110",
            "Run 117", "Run 118", "Run 125", "Run 151", "Run 152")):
        print(p180[:650])

print("\nCPU WEIGHT CACHE INTERFACE:", w63.__code__.co_names)
with open("/work/notes.md", "a") as f180:
    f180.write("\n## Runs 179–180 / reject final-layer-only fit\n"
               "Closed-form final value-layer candidates could not attain accepted effects "
               "within factual +0.005 NLL. Lambda 1 gave validation +0.00184 but slopes "
               "only I~−1.03/NE~+0.39 at x2; stronger edits violated fit. No weights saved. "
               "Captured all dose contrasts before the value GELU to diagnose the private wire.\n")
```