The recovered probe measures a continuous embedding perturbation, not a real dose edit, so I won’t treat it as evidence of a dose response. I’ll first test a small, training-only calibration of reading persistence and dispersion. It can be expressed in transition weights. It is useful only if its factual improvement generalizes; any dose effects must then be rechecked.

```python
# Closed-form moment calibration: six coefficients per reading code.
# No gradients through the world model and no weight optimizer.
r224_basis = {}
r224_prev_basis = {}
r224_events = sorted(c210_train["meta"]["event"].unique())

for event224 in r224_events:
    mask224 = c210_train["meta"]["event"].to_numpy() == event224
    centres224 = np.asarray(c210_train["centres"][np.flatnonzero(mask224)[0]], dtype=float)
    scale224 = max(float(np.quantile(centres224[1:], .75)
                         - np.quantile(centres224[1:], .25)), 1.0)
    z224 = np.clip((centres224 - np.median(centres224[1:])) / scale224, -4, 4)
    prev224 = z224.copy()
    prev224[0] = 0.0
    table224 = np.empty((17, 17, 6), dtype=np.float64)
    for j224 in range(17):
        u224 = prev224[j224]
        table224[j224] = np.column_stack([
            z224, z224*z224,
            z224*u224, z224*(u224*u224),
            z224*z224*u224, z224*z224*(u224*u224)
        ])
    r224_basis[event224] = table224

def r224_calibration(weights, ridge, fit_mask):
    pp = c211_logits(weights, c210_train).softmax(-1).mean(0).numpy().astype(float)
    yy = np.asarray(c210_train["y"])
    mm = c210_train["meta"]
    delta = np.zeros_like(np.asarray(r217_T0), dtype=np.float32)
    coefs = {}
    train_rows = np.asarray(r217_cache_rows["train"])
    for event in r224_events:
        mask = (mm["event"].to_numpy() == event) & fit_mask
        jj = mm["last_bin"].to_numpy(dtype=int)[mask]
        ff = r224_basis[event][jj]             # readings x bins x coefficients
        p = pp[mask]
        expected = np.einsum("nb,nbk->nk", p, ff)
        score = (ff[np.arange(mask.sum()), yy[mask]] - expected).mean(0)
        second = np.einsum("nb,nbk,nbl->kl", p, ff, ff) / mask.sum()
        fisher = second - expected.T @ expected / mask.sum()
        penalty = ridge * np.diag(np.maximum(np.diag(fisher), 0.05))
        beta = np.linalg.solve(fisher + penalty, score)
        coefs[event] = beta
        event_indices = np.flatnonzero(mm["event"].to_numpy() == event)
        row0 = int(train_rows[event_indices[0]] - mm.iloc[event_indices[0]]["last_bin"])
        delta[row0:row0+17] = np.einsum("jbk,k->jb", r224_basis[event], beta)
    return delta, coefs

r224_fit_mask = np.isin(c210_train["meta"]["stay_id"].to_numpy(), r137_fit_ids)
r224_val_mask = ~r224_fit_mask
r224_results = []
r224_calibrations = {}

for candidate224 in ["original", "combined_strong"]:
    weights224 = c206_candidates[candidate224]
    for ridge224 in [0.02, 0.1, 0.5, 2.0]:
        delta224, coef224 = r224_calibration(weights224, ridge224, r224_fit_mask)
        r224_calibrations[candidate224, ridge224] = (delta224, coef224)
        for strength224 in [0.5, 1.0]:
            trloss224, _ = r217_eval(weights224, c210_train,
                                    strength224 * delta224, "train")
            tuloss224, _ = r217_eval(weights224, c210_tune,
                                    strength224 * delta224, "tune100")
            trdiff224 = trloss224 - np.asarray(c210_train["loss0"])
            tudiff224 = tuloss224 - np.asarray(c210_tune["loss0"])
            r224_results.append({
                "candidate": candidate224, "ridge": ridge224, "scale": strength224,
                "train_fit": float(trdiff224[r224_fit_mask].mean()),
                "train_validation": float(trdiff224[r224_val_mask].mean()),
                "tune100": float(tudiff224.mean())
            })

print(pd.DataFrame(r224_results).round(6).to_string(index=False))
print("No weights installed.")
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 224: low-dimensional transition calibration screen\n")
    f.write("Exposure guard was refuted: essentially all correction cost follows drug exposure. "
            "Screened six-parameter per-code persistence/dispersion calibrators fitted "
            "by training-only moment equations; no weights changed.\n")
    f.write(pd.DataFrame(r224_results).to_string(index=False) + "\n")
```
