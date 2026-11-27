Only attention head 2 carries substantial dose information, and neither feature fit is yet accurate enough to use uncalibrated. I’ll check the average response at each dose multiplier and prepare a readout probe. This will show whether the errors are mainly under-response—which we can calibrate—or inconsistent nonlinear responses that would make an edit unreliable.

```python
import numpy as np

chosen90 = {
    "hidden_l1": contrast_fits88[(1.0, 1e-6)],
    "hidden_l10": contrast_fits88[(10.0, 1e-6)],
}
offset90 = 0
for meta90, dx90 in zip(contrast_meta88, dose_design88):
    drug90, factor90, sel90 = meta90
    n90 = len(sel90)*4
    held90 = ~np.repeat(train88[sel90], 4)
    means90 = {}
    for name90, c90 in chosen90.items():
        means90[name90] = np.mean(dx90[held90]@c90, axis=0).round(4).tolist()
    print(drug90, factor90,
          "target", np.mean(DY88[offset90:offset90+n90][held90], axis=0).round(4),
          "predicted", means90)
    offset90 += n90

print("FACTUAL SCORES BY BUNDLE")
score90 = (fact_norm88@chosen90["hidden_l10"]).reshape(len(idx81), 4, 2).mean(axis=1)
for label90, mask90 in [("drug", active81), ("non-drug", ~active81)]:
    print(label90, "mean", score90[mask90].mean(axis=0),
          "SD", score90[mask90].std(axis=0),
          "quantiles", np.quantile(score90[mask90], [.01,.5,.99], axis=0))

print("CENTRE TABLE", repr(VOCAB["value_bin_centres"])[:700])
print("NUMERIC NOTE", VOCAB["numeric_note"])
print("VALUE NOTE", VOCAB["value_bin_note"])
print("HEAD SUPPORT TENSORS")
for n90 in NAMES:
    if n90.startswith(("embed_transition.value", "z_prior", "embed.bins.value")):
        print(n90, SHAPES[n90])
print("LAST NUMERIC COLUMN NORMS", W["last_numeric.weight"].norm(dim=0).numpy())

read_codes90 = ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff")
last_values90 = np.full((len(r40_info), 4), np.nan)
last_hours90 = np.full((len(r40_info), 4), np.nan)
registers90 = {}
for i90, bundle90 in enumerate(r40_info):
    sid90 = int(S["stay_id"][i90])
    reg90 = registers90.setdefault(sid90, {})
    for row90, end90 in bundle90:
        if not end90 and row90.event in read_codes90:
            reg90[row90.event] = (float(row90.value), float(row90.hour))
    for j90, code90 in enumerate(read_codes90):
        if code90 in reg90:
            last_values90[i90, j90], last_hours90[i90, j90] = reg90[code90]
print("REGISTERS", last_values90.shape,
      "available", np.sum(np.isfinite(last_values90), axis=0))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 90; forwards 7. Attention dose sensitivity is concentrated in head 2. Comparing held-out average dose contrasts across factors and inactive factual scores; preparing actual preceding-reading registers for a finite-difference readout probe. No edits.\n")
```