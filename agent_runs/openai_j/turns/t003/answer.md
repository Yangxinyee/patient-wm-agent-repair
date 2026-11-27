Doses start five minutes after each hour’s readings, giving a clear pre-treatment and next-hour ordering. Both drugs vary substantially, but that alone does not establish causality. I’ll compare next-hour outcomes after controlling for observed history; consistent estimates across sensors would support a drug effect, while sensitivity to controls would weaken that conclusion.

```python
from scipy import linalg

MEAS = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
DRUGS = ["insulin", "norepinephrine"]
HOURS = np.arange(-1, 25)

def make_panel(df):
    d = df.copy()
    d["t"] = np.floor(d.hour + 1e-6).astype(np.int16)
    d["x"] = np.where(d.event.isin(DRUGS), d.rate, d.value)
    ids = np.sort(d.stay_id.unique())
    index = pd.MultiIndex.from_product([ids, HOURS], names=["stay_id", "t"])
    p = d[d.event.isin(MEAS + DRUGS)].pivot_table(
        index=["stay_id", "t"], columns="event", values="x",
        aggfunc="mean", observed=True
    ).reindex(index).reindex(columns=MEAS + DRUGS)
    p[DRUGS] = p[DRUGS].fillna(0)
    return ids, p

train_ids, panel = make_panel(train)
tuning_ids, tuning_panel = make_panel(tuning)
Y = panel[MEAS].to_numpy().reshape(len(train_ids), 26, 4)
A = panel[DRUGS].to_numpy().reshape(len(train_ids), 26, 2)

print("=== SIMULTANEOUS SENSOR DIFFERENCES ===")
for j, k in [(0, 1), (2, 3)]:
    diff = Y[:, :, j] - Y[:, :, k]
    print(MEAS[j], "-", MEAS[k],
          "n =", np.isfinite(diff).sum(),
          "mean =", round(np.nanmean(diff), 4),
          "sd =", round(np.nanstd(diff), 4),
          "quantiles =", np.nanquantile(diff, [.05, .5, .95]))

ff = panel[MEAS].groupby(level=0, sort=False).ffill()
F = ff.to_numpy().reshape(len(train_ids), 26, 4)
med = np.nanmedian(Y, axis=(0, 1))
F = np.where(np.isfinite(F), F, med[None, None, :])
seen = np.isfinite(ff.to_numpy()).reshape(len(train_ids), 26, 4)
last_t = np.where(np.isfinite(Y), HOURS[None, :, None], np.nan)
last_t = pd.DataFrame(last_t.reshape(-1, 4), index=panel.index).groupby(
    level=0, sort=False
).ffill().to_numpy().reshape(len(train_ids), 26, 4)
age = np.where(np.isfinite(last_t), HOURS[None, :, None] - last_t, 30)

# Each row: pre-dose history at hour t; response at hour t+1.
cur = slice(1, 25)
prev1 = F[:, :24, :]
prev2 = np.concatenate([F[:, :1, :], F[:, :23, :]], axis=1)
controls = np.concatenate([
    F[:, cur, :], prev1, prev2,
    np.broadcast_to(F[:, :1, :], (len(train_ids), 24, 4)),
    np.isfinite(Y[:, cur, :]).astype(float),
    seen[:, cur, :].astype(float),
    age[:, cur, :],
    A[:, :24, :],
    np.concatenate([A[:, :1, :], A[:, :23, :]], axis=1),
    np.broadcast_to(np.arange(24)[None, :, None], (len(train_ids), 24, 1)),
    np.broadcast_to((np.arange(24)**2)[None, :, None], (len(train_ids), 24, 1)),
], axis=2).reshape(-1, 34)
C_mean = controls.mean(axis=0)
C_sd = controls.std(axis=0)
C_sd[C_sd < 1e-8] = 1
C = (controls - C_mean) / C_sd
dose = A[:, cur, :].reshape(-1, 2)
presence = (dose > 0).astype(float)
stay_row = np.repeat(np.arange(len(train_ids)), 24)
next_y = Y[:, 2:26, :].reshape(-1, 4)
current_y = Y[:, cur, :].reshape(-1, 4)

def cluster_ols(X, target, groups):
    xtx = X.T @ X
    inv = linalg.inv(xtx + np.eye(X.shape[1]) * 1e-6)
    beta = inv @ (X.T @ target)
    resid = target - X @ beta
    scores = np.zeros((len(train_ids), X.shape[1]))
    np.add.at(scores, groups, X * resid[:, None])
    cov = inv @ (scores.T @ scores) @ inv
    return beta, np.sqrt(np.maximum(np.diag(cov), 0)), np.std(resid)

print("\n=== NEXT-HOUR CONDITIONAL ASSOCIATIONS (NOT YET CAUSAL) ===")
association_results = []
for sensor in range(4):
    mask = np.isfinite(next_y[:, sensor]) & np.isfinite(current_y[:, sensor])
    for label, control_subset in [
        ("current readings", C[:, :4]),
        ("full past history", C),
    ]:
        X = np.column_stack([np.ones(mask.sum()), dose[mask],
                             presence[mask], control_subset[mask]])
        b, se, rsd = cluster_ols(X, next_y[mask, sensor], stay_row[mask])
        row = {"sensor": MEAS[sensor], "controls": label, "n": int(mask.sum()),
               "insulin": b[1], "insulin_se": se[1],
               "norepinephrine": b[2], "norepinephrine_se": se[2],
               "residual_sd": rsd}
        association_results.append(row)
print(pd.DataFrame(association_results).round(5).to_string(index=False))

print("\n=== DOSE DISTRIBUTION ===")
for j, drug in enumerate(DRUGS):
    r = dose[:, j]
    print(drug, "fraction treated:", round((r > 0).mean(), 4),
          "treated quantiles:", np.quantile(r[r > 0], [0, .1, .5, .9, 1]),
          "unique positive rates:", len(np.unique(r[r > 0])))

with open("/work/notes.md", "a") as f:
    f.write("\n## Hourly panel and preliminary associations\n")
    f.write("- Infusions begin at hour t+0.0833, after hour-t measurements.\n")
    f.write(pd.DataFrame(association_results).to_string(index=False) + "\n")
    f.write("- These are conditional associations, not established causal effects.\n")
    f.write("- Python runs: 3. Forward passes: 0. No edits yet.\n")
```