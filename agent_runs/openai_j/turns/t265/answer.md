Halving and doubling often move the inputs in similar directions, and their current correction scores overlap substantially with factual scores. A simple threshold is therefore unlikely to suffice. I’ll first look for a doubling-sensitive direction that leaves halving sensitivity unchanged, using training stays only. It must retain that separation on held-out stays before I consider a nonlinear weight edit.

```python
# Construct a statistical probe, not a model-weight edit.
mapfit235 = np.asarray(fitall208, dtype=bool) & np.asarray(mapmask208, dtype=bool)
x235 = wm_train145["inputs"].numpy()
center235 = x235[:, mapfit235].mean(axis=(0, 1)).astype(np.float64)
scale235 = np.maximum(x235[:, mapfit235].std(axis=(0, 1)), 0.02).astype(np.float64)
xf235 = ((x235[:, mapfit235] - center235) / scale235).reshape(-1, 192)
cov235 = xf235.T @ xf235 / len(xf235)

def contrast235(drug, factor, event, subset):
    mc = wm_cf146[(drug, factor)]
    mask = (np.asarray(events206 == event) &
            np.asarray(mc["active"], dtype=bool) &
            np.asarray(subset, dtype=bool))
    rate = np.asarray(mc["delta_rate"], dtype=np.float64)[mask]
    dx = (mc["inputs"] - wm_train145["inputs"]).mean(0).numpy()[mask] / scale235
    return (dx * rate[:, None]).sum(0) / max(np.square(rate).sum(), 1e-12)

m2_235 = contrast235("norepinephrine", 2.0, "map_arterial", fitall208)
constraints235 = np.stack([
    contrast235("norepinephrine", 0.5, event, fitall208)
    for event in ["map_arterial", "map_cuff"]
])

directions235 = {}
for ridge in [0.01, 0.03, 0.1, 0.3, 1.0]:
    inv_m = np.linalg.solve(cov235 + ridge*np.eye(192), m2_235)
    inv_c = np.linalg.solve(cov235 + ridge*np.eye(192), constraints235.T)
    correction = np.linalg.solve(
        constraints235 @ inv_c + 1e-8*np.eye(len(constraints235)),
        constraints235 @ inv_m)
    direction = inv_m - inv_c @ correction
    sd = np.sqrt(max(direction @ cov235 @ direction, 1e-12))
    direction /= sd
    raw = direction / scale235
    affine = np.r_[raw, -center235 @ raw]
    directions235[ridge] = affine
    rows = []
    for subset_name, subset in [("fit", fitall208), ("heldout", cvall208)]:
        for factor in [0.5, 2.0]:
            for event in ["map_arterial", "map_cuff"]:
                rows.append((subset_name, factor, event,
                             contrast235("norepinephrine", factor, event, subset)
                             @ direction))
    print("RIDGE", ridge)
    print(pd.DataFrame(rows, columns=["subset", "factor", "sensor",
                                     "score_change_per_rate"]).round(4).to_string(index=False))

print("\nCURRENT LOCAL FACTUAL LOSS")
print("fit", float(np.mean(pred233["nll"][fitall208]
                          - wm_train145["nll"].mean(0).numpy()[fitall208])))
print("heldout", float(np.mean(pred233["nll"][cvall208]
                              - wm_train145["nll"].mean(0).numpy()[cvall208])))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 233–235: MAP input-geometry audit\n"
            "- Verified the live MAP coefficient equals the gain-8 candidate.\n"
            "- Half/double input contrasts are frequently not opposite; median "
            "cosines are +0.34 arterial and +0.26 cuff, consistent with nonlinear "
            "dose encoding rather than a simple signed rate channel.\n"
            "- Current factual/counterfactual scalar scores overlap substantially.\n"
            "- Constructed doubling-sensitive statistical directions with halving "
            "constraints, checking held-out stays before any nonlinear edit.\n"
            "- No model weights changed.\n")
```