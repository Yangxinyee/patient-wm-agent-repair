The original factual loss is 1.63805 nats per reading on these training stays. That alone says nothing about causal accuracy. I’ll now measure whether its prediction errors follow dose innovations and test pre/post-action state contrasts. Consistent residual slopes and recoverable action changes would support a centered repair; noisy or negligible results would caution against choosing its size from this small sample.

```python
from scipy import stats

train_factual0 = pd.read_csv("/work/fwd_4/factual_original.csv")
print("=== NON-TRANSFORMER TENSOR NAMES ===")
for nm in NAMES:
    if not any(t in nm for t in ("core.prelude.", "core.blocks.", "core.coda.")):
        print(nm, SHAPES[nm])
print("Event kinds:", VOCAB["event_kinds"])

def cluster_ols_diagnostic(x, y, groups):
    x = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
    y = np.asarray(y, dtype=float)
    good = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y, groups = x[good], y[good], np.asarray(groups)[good]
    inv = np.linalg.pinv(x.T @ x)
    coef = inv @ x.T @ y
    _, g = np.unique(groups, return_inverse=True)
    score = np.zeros((g.max() + 1, x.shape[1]))
    np.add.at(score, g, x * (y - x @ coef)[:, None])
    cov = inv @ (score.T @ score) @ inv
    cov *= len(score) / max(len(score) - 1, 1)
    return coef[1:], np.sqrt(np.diag(cov))[1:], len(y)

factual_missing_effect0 = {}
print("\n=== FACTUAL ERROR VS PRECEDING DOSE INNOVATION ===")
for ev in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    q = train_factual0[
        (train_factual0.event == ev) &
        (train_factual0.hour >= 1) & (train_factual0.hour <= 24)
    ].copy()
    pos = P.index.get_indexer(pd.MultiIndex.from_arrays([
        q.stay_id.to_numpy(), np.rint(q.hour.to_numpy()).astype(int) - 1
    ]))
    assert np.all(pos >= 0)
    cp = causal_position[pos]
    assert np.all(cp >= 0)
    beta_miss, se_miss, n_miss = cluster_ols_diagnostic(
        dose_resid[cp],
        q.observed.to_numpy() - q.predicted_mean.to_numpy(),
        q.stay_id.to_numpy()
    )
    factual_missing_effect0[ev] = {
        "beta": beta_miss.tolist(), "se": se_miss.tolist(), "n": n_miss
    }
    print(ev, "n", n_miss, "missing insulin / NE slopes:",
          np.round(beta_miss, 3), "cluster SE:", np.round(se_miss, 3))

print("\n=== PRE/POST-ACTION INNOVATION RECOVERY ===")
dr = state_drug_rows0
assert np.all(train_states0["stay_id"][dr] == train_states0["stay_id"][dr - 1])
sx = train_states0["x"]
prepost_features0 = np.column_stack([sx[dr, :192], sx[dr - 1, :192]])
contrast_features0 = np.column_stack([
    sx[dr, :192] - sx[dr - 1, :192],
    sx[dr - 1, :192],
    sx[dr, 192:960].reshape(-1, 4, 192).mean(axis=1) -
        sx[dr - 1, 192:960].reshape(-1, 4, 192).mean(axis=1)
])
contrast_recovery0 = {}
for label, features in [("pre+post out", prepost_features0),
                         ("action contrasts", contrast_features0)]:
    for alpha in (10.0, 100.0, 1000.0):
        estimate = np.zeros_like(state_drug_innov0, dtype=float)
        for f in (0, 1):
            keep = state_fold0 != f
            reg = Ridge(alpha=alpha).fit(features[keep], state_drug_innov0[keep])
            estimate[~keep] = reg.predict(features[~keep])
        score = r2_score(state_drug_innov0, estimate, multioutput="raw_values")
        contrast_recovery0[(label, alpha)] = score.tolist()
        print(label, "ridge", alpha, "R2:", np.round(score, 4))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 10 / forward 4: factual residual and action-contrast diagnostics\n"
        "- Original train-100 factual value NLL: 1.63805 over 4,510 readings.\n"
        "- Factual-error innovation slopes: " + repr(factual_missing_effect0) + "\n"
        "- Pre/post-action innovation recovery: " + repr(contrast_recovery0) + "\n"
        "- No weights edited. These small-sample diagnostics do not replace full-record causal estimates.\n"
        "- Python runs: 10; forward passes: 4.\n"
    )
```