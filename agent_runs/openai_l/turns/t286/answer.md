The average sensor errors are too small to explain the factual penalty; the distortion varies within each sensor. I’ll test a regularized correction conditioned on the previous reading’s bin, fitted only on training states. Lower NLL on independent states would support it; dose rollouts will still need validation.

```python
transition_name253 = next(n for n in NAMES
                          if n.startswith("embed_transition.value_bin")
                          and n.endswith("weight"))
transition_original253 = np.asarray(load_original(transition_name253)).copy()
codes253 = VOCAB["codes"]
print("Transition:", transition_name253, transition_original253.shape)
print("Codes:", codes253)

def code_id253(event):
    if isinstance(codes253, dict):
        v = codes253[event]
        return int(v["id"] if isinstance(v, dict) else v)
    return list(codes253).index(event)

def recover_lastbins253(events, offsets):
    bins = np.zeros(len(events), dtype=int)
    worst = 0.0
    for event in np.unique(events):
        use = np.asarray(events) == event
        cid = code_id253(event)
        rows = transition_original253[cid * 17:(cid + 1) * 17]
        distance = np.mean((offsets[use, None, :] - rows[None, :, :])**2, axis=-1)
        bins[use] = distance.argmin(axis=1)
        worst = max(worst, float(np.sqrt(distance.min(axis=1).max())))
    return bins, worst

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

        ix_train253 = np.array([code_id253(e) for e in events179]) * 17 + lastbins_train253
        ix_tune253 = np.array([code_id253(e) for e in events225]) * 17 + lastbins_tune253
        dc_train253 = correction253[ix_train253]
        dc_tune253 = correction253[ix_tune253]
        delta_train253 = nll179(Lt253 + dc_train253[None]) - base_nll_train234
        delta_tune253 = nll228(Lq253 + dc_tune253[None], B225) - base_nll_tune234
        score253 = (float(delta_train253[fitmask179].mean()),
                    float(delta_train253[valmask179].mean()),
                    float(delta_tune253.mean()))
        calibrations253[(alpha253, smoothing253)] = {
            "correction": correction253, "score": score253}
        print("Conditional-offset candidate", (alpha253, smoothing253),
              "NLL deltas train/validation/tuning:", score253,
              "offset RMS:", float(np.sqrt(np.mean(correction253**2))))
else:
    print("Offsets are not exact transition-table rows; no calibration fitted.")
```