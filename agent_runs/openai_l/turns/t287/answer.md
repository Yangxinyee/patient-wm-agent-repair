The attempt stopped before fitting because the vocabulary stores code IDs in structured entries. No weights changed. I’ll fix that lookup and rerun the same test. The correction is worth pursuing only if it reduces factual distortion beyond the states used to fit it.

```python
code_lookup254 = {c["name"]: int(c["id"]) for c in VOCAB["codes"]}
def code_id253(event):
    return code_lookup254[event]

lastbins_train253, error_train253 = recover_lastbins253(events179, offset204[0])
lastbins_tune253, error_tune253 = recover_lastbins253(events225, offset234[0])
print("Register reconstruction RMS maxima:", error_train253, error_tune253)

calibrations253 = {}
if max(error_train253, error_tune253) < 1e-4:
    Lt253 = arch235["F"] @ Tlive252 + offset204
    Lq253 = stats234[("pair4", .1)]["Ftune"] @ Tlive252 + offset234
    D253 = np.diff(np.eye(16), n=2, axis=0)
    smooth253 = np.zeros((17, 17))
    smooth253[1:, 1:] = D253.T @ D253
    ix_train253 = np.array([code_id253(e) for e in events179]) * 17 + lastbins_train253
    ix_tune253 = np.array([code_id253(e) for e in events225]) * 17 + lastbins_tune253

    for alpha253, smoothing253 in [(10., 0.), (30., 0.), (100., 0.),
                                   (10., 30.), (30., 100.), (100., 100.)]:
        correction253 = np.zeros_like(transition_original253, dtype=np.float64)
        for event253 in np.unique(events179):
            use253 = (events179 == event253) & fitmask179
            lb253 = lastbins_train253[use253]
            rr253 = residual252[:, use253].mean(axis=0)
            counts253 = np.bincount(lb253, minlength=17).astype(float)
            sums253 = np.zeros((17, 17))
            np.add.at(sums253, lb253, rr253)
            fit253 = np.linalg.solve(
                np.diag(counts253 + alpha253) + smoothing253 * smooth253,
                sums253)
            fit253 -= fit253.mean(axis=1, keepdims=True)
            cid253 = code_id253(event253)
            correction253[cid253 * 17:(cid253 + 1) * 17] = fit253

        dc_train253 = correction253[ix_train253]
        dc_tune253 = correction253[ix_tune253]
        delta_train253 = nll179(Lt253 + dc_train253[None]) - base_nll_train234
        delta_tune253 = nll228(Lq253 + dc_tune253[None], B225) - base_nll_tune234
        score253 = (float(delta_train253[fitmask179].mean()),
                    float(delta_train253[valmask179].mean()),
                    float(delta_tune253.mean()))
        calibrations253[(alpha253, smoothing253)] = {
            "correction": correction253, "score": score253}
        print((alpha253, smoothing253), "NLL deltas:", score253,
              "offset RMS:", float(np.sqrt(np.mean(correction253**2))))
else:
    print("No calibration fitted: transition-row matching failed.")

print("=== CLOSED-FORM HEAD REPRODUCTION ===")
st254 = stats234[("pair4", .1)]
print("Contrast keys:", list(st254["GD"]), list(st254["RD"]))
A254 = (st254["GF"] + .5 * st254["GD"]["insulin"]
        + .4 * st254["GD"]["norepinephrine"] + ridge242 * np.eye(193))
R254 = (st254["RF"] + .5 * st254["RD"]["insulin"]
        + .4 * st254["RD"]["norepinephrine"])
T254 = st254["T0"] + np.linalg.solve(A254, R254) / st254["scale"][:, None]
print("Difference from saved .5/.4 head:",
      float(np.max(np.abs(T254 - grid242[(.5, .4)]["T"]))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 252–254 / conditional factual compensation\n"
            "Constant sensor errors are small; testing regularized "
            "original-logit residuals conditioned on prior sensor bin, "
            "fitted on training fitmask only. Run253 stopped on structured "
            "vocabulary lookup before fitting; run254 corrects it. "
            "No weights changed.\n")
```