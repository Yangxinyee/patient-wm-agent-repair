With richer histories, the estimates remain close to −2 glucose units per insulin-rate unit and +3.5 to +3.9 MAP units per norepinephrine-rate unit. Agreement across MAP sensors is encouraging, though it does not prove that all confounding is removed. I’ll estimate how effects persist over time and translate them into rollout targets; inconsistent persistence estimates would make those targets unreliable.

```python
from sklearn.linear_model import Ridge

def sensor_iv_persistence(gold, proxy, drug, drug_effect):
    gg = panel.groupby(level=0, sort=False)
    next_y = gg[gold].shift(-1) - drug_effect * panel[drug]
    mask = (panel[gold].notna() & panel[proxy].notna() & next_y.notna() &
            F.hour.between(1, 23))
    rows = np.flatnonzero(mask.to_numpy())
    ctrl_names = [c for c in F.columns
                  if ("lag" in c or "initial" in c or c == "hour")
                  and not c.startswith(drug)]
    C = F[ctrl_names].iloc[rows].copy()
    for c in list(C.columns):
        C[c + "_missing"] = C[c].isna().astype(float)
        C[c] = C[c].fillna(C[c].median())
    C = np.column_stack([np.ones(len(rows)), C.to_numpy(float)])
    target = np.column_stack([
        panel[gold].iloc[rows].to_numpy(float),
        panel[proxy].iloc[rows].to_numpy(float),
        next_y.iloc[rows].to_numpy(float)
    ])
    residual = target - C @ np.linalg.lstsq(C, target, rcond=None)[0]
    rg, rp, ry = residual.T
    den = rp @ rg
    a = (rp @ ry) / den
    ids = panel.index.get_level_values(0).to_numpy()[rows]
    _, groups = np.unique(ids, return_inverse=True)
    scores = np.bincount(groups, weights=rp * (ry - a * rg)) / den
    return dict(drug=drug, n=len(rows), assumed_effect=drug_effect,
                persistence=float(a), cluster_se=float(np.sqrt(scores @ scores)),
                cross_sensor_residual_cov=float(np.mean(rp * rg)))

persist_results = []
for b in [-1.55, -1.8, -2.0]:
    persist_results.append(sensor_iv_persistence(
        "glucose_lab", "glucose_finger", "insulin", b))
for b in [3.0, 3.6, 4.0]:
    persist_results.append(sensor_iv_persistence(
        "map_cuff", "map_arterial", "norepinephrine", b))
print("=== MEASUREMENT-ERROR-ADJUSTED PERSISTENCE ===")
print(pd.DataFrame(persist_results).round(5).to_string(index=False))

def linear_rollout_targets(data_panel, drug, b, a):
    arr = data_panel[drug].unstack("t").reindex(columns=range(0, 25)).fillna(0)
    delta = np.zeros(len(arr))
    reached = np.zeros(len(arr), bool)
    sums = np.zeros(len(arr))
    counts = np.zeros(len(arr))
    for h in range(12, 24):
        dose = arr[h].to_numpy()
        reached = reached | (dose > 0)
        delta = a * delta + b * dose
        sums += np.where(reached, delta, 0)
        counts += reached
    means = sums[counts > 0] / counts[counts > 0]
    return dict(drug=drug, b=b, a=a, eligible_stays=len(means),
                x2=float(means.mean()), half=float(-.5 * means.mean()))

target_ranges = []
for drug, effects, persistence_range in [
    ("insulin", [-1.55, -1.8, -2.0], [.8, .9, .95]),
    ("norepinephrine", [3.0, 3.6, 4.0], [.7, .8, .9])
]:
    for b in effects:
        for a in persistence_range:
            target_ranges.append(linear_rollout_targets(tune_panel, drug, b, a))
print("\n=== CONDITIONAL LINEAR ROLLOUT TARGETS, NOT OBSERVED TRUTH ===")
print(pd.DataFrame(target_ranges).round(4).to_string(index=False))

print("\n=== NORMALISATION NOTES ===")
print(VOCAB.get("numeric_note"))
print("event kinds:", VOCAB.get("event_kinds"))
print("atom kinds:", VOCAB.get("atom_kinds"))
print("flow:", [(n, SHAPES[n]) for n in NAMES if "flow" in n])
print("embed other:", [(n, SHAPES[n]) for n in NAMES
                       if n.startswith("embed.") and not any(
                           x in n for x in ["categorical", "bins", "numeric", "code"])])
print("last numeric shape:", SHAPES["last_numeric.weight"])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 12: persistence and rollout target sensitivity\n")
    f.write(pd.DataFrame(persist_results).round(5).to_string(index=False) + "\n")
    f.write(pd.DataFrame(target_ranges).round(4).to_string(index=False) + "\n")
    f.write("Targets assume linear, time-homogeneous drug effects; they are estimates, not tool-provided truth.\n")
```
