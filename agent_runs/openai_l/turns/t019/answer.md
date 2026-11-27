The audit confirms that the cross-fitted estimates use the correct action-to-next-reading timing. Corrected subsets still support insulin near −2 and norepinephrine near +3, though cuff estimates are less consistent. I’ll now test dosing discontinuities as an independent check, including pre-dose readings and monitoring flags; jumps in those placebos would weaken the causal interpretation.

```python
def rd16(running, treatment, outcome, cutoff, width, extra_mask=None):
    r = running.reshape(-1).astype(np.float64) - cutoff
    a = treatment.reshape(-1).astype(np.float64)
    y = outcome.reshape(-1).astype(np.float64)
    ok = np.isfinite(r) & np.isfinite(a) & np.isfinite(y) & (np.abs(r) <= width)
    if extra_mask is not None:
        ok &= extra_mask.reshape(-1)
    rr = r[ok]
    zz = (rr > 0).astype(float)
    xx = np.column_stack([np.ones(ok.sum()), rr, zz, zz * rr,
                          np.tile(np.arange(24), 30000)[ok] / 24])
    inv = np.linalg.pinv(xx.T @ xx)
    ba = inv @ (xx.T @ a[ok])
    by = inv @ (xx.T @ y[ok])
    ja, jy = ba[2], by[2]
    if abs(ja) < 1e-8:
        return (int(ok.sum()), ja, jy, np.nan, np.nan)
    b = jy / ja
    err = (y[ok] - xx @ by) - b * (a[ok] - xx @ ba)
    influence = (xx @ inv[:, 2]) * err / ja
    grouped = np.bincount(groups14[ok], weights=influence, minlength=30000)
    return (int(ok.sum()), float(ja), float(jy), float(b),
            float(np.sqrt(grouped @ grouped)))

rd16_results = []
for cutoff16 in [179.5, 219.5]:
    for width16 in [6, 12]:
        for sensor16 in ["glucose_finger", "glucose_lab"]:
            rr16 = rd16(P["glucose_finger"][:, :24], A["insulin"][:, :24],
                        P[sensor16][:, 1:25], cutoff16, width16)
            rd16_results.append(("I", cutoff16, width16, sensor16, rr16))
            print("I", cutoff16, width16, sensor16, np.round(rr16, 4))
        for target16, yy16 in [
            ("pre-dose lab", P["glucose_lab"][:, :24]),
            ("next lab observed", np.isfinite(P["glucose_lab"][:, 1:25]).astype(float))
        ]:
            rr16 = rd16(P["glucose_finger"][:, :24], A["insulin"][:, :24],
                        yy16, cutoff16, width16)
            print("I placebo", cutoff16, width16, target16, np.round(rr16, 4))

both16 = np.isfinite(P["map_arterial"][:, :24]) & np.isfinite(P["map_cuff"][:, :24])
for source16 in ["map_arterial", "map_cuff"]:
    for width16 in [6, 12]:
        for sensor16 in ["map_arterial", "map_cuff"]:
            rr16 = rd16(P[source16][:, :24], A["norepinephrine"][:, :24],
                        P[sensor16][:, 1:25], 64.5, width16)
            rd16_results.append(("NE", source16, width16, sensor16, rr16))
            print("NE", source16, width16, sensor16, np.round(rr16, 4))
        other16 = "map_cuff" if source16 == "map_arterial" else "map_arterial"
        rr16 = rd16(P[source16][:, :24], A["norepinephrine"][:, :24],
                    P[other16][:, :24], 64.5, width16, both16)
        print("NE placebo", source16, width16, "pre-dose other", np.round(rr16, 4))
        rr16 = rd16(P[source16][:, :24], A["norepinephrine"][:, :24],
                    np.isfinite(P["map_arterial"][:, 1:25]).astype(float), 64.5, width16)
        print("NE placebo", source16, width16, "next arterial observed", np.round(rr16, 4))

print("\nCurrent observed MAP versus current NE:")
for source16 in ["map_arterial", "map_cuff"]:
    pp16 = pd.DataFrame({
        "MAP": P[source16][:, :24].reshape(-1),
        "rate": A["norepinephrine"][:, :24].reshape(-1)
    })
    print(source16)
    print(pp16.groupby("MAP").rate.agg(["size", "mean", "median"]).loc[60:70].round(4).to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 16\nAlignment audit reconstructed dose at h and reading at h+1 exactly. Corrected DML: finger I=-1.936(.045), lab=-1.968(.079), arterial NE=3.049(.088), cuff=2.529(.172); arterial active/both-current=3.150(.158), cuff active/both-current=3.297(.428). Running independent insulin/NE threshold and pre-dose/monitoring-placebo checks. No edits.\n")
```
