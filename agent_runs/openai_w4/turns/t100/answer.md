The raw inputs recover insulin innovations substantially better than the compressed hidden layer, and their half- and double-dose responses have the right direction. Norepinephrine is less cleanly encoded. I will test gates across all four dose edits and estimate sustained targets by convolving record-derived lag responses with recorded doses. Gate leakage or unstable target estimates would argue against proceeding.

```python
from sklearn.svm import LinearSVC
from scipy.optimize import least_squares

gate_models88, gate_rows88 = {}, []
u_sets88 = [raw87[:, 384:]] + [a[:, 384:] for a in raw_cf87.values()]
u_train88 = np.concatenate([a[fit82] for a in u_sets88])
for drug88 in ["insulin", "norepinephrine"]:
    dm88 = repair_bank["has_" + drug88][idx82].astype(bool)
    yy88 = np.tile(dm88[fit82], len(u_sets88))
    gate88 = LinearSVC(C=1.0, dual=False, max_iter=10000,
                      tol=1e-6, class_weight="balanced").fit(u_train88, yy88)
    gate_models88[drug88] = gate88
    for label88, inputs88 in [("factual", raw87)] + list(raw_cf87.items()):
        margin88 = gate88.decision_function(inputs88[test82, 384:])
        active88 = dm88[test82]
        gate_rows88.append({
            "drug": drug88, "input": str(label88),
            "accuracy": np.mean((margin88 > 0) == active88),
            "active_min": margin88[active88].min(),
            "active_q05": np.quantile(margin88[active88], .05),
            "inactive_max": margin88[~active88].max()
        })
print("ACTION GATES\n", pd.DataFrame(gate_rows88).round(4).to_string(index=False))

def two_pole_kernel88(params, length):
    amplitude, pole1, pole2 = params
    return amplitude * np.array([
        sum(pole1**j * pole2**(lag-j) for j in range(lag+1))
        for lag in range(length)])

kernel_models88, target_rows88 = {}, []
for drug88 in ["insulin", "norepinephrine"]:
    empirical88 = np.asarray(provisional_kernels[drug88])
    sign88 = -1.0 if drug88 == "insulin" else 1.0
    result88 = least_squares(
        lambda p: two_pole_kernel88([sign88*p[0], p[1], p[2]], 6) - empirical88,
        x0=[abs(empirical88[0]), .6 if sign88 < 0 else .5,
            .6 if sign88 < 0 else .3],
        bounds=([0, 0, 0], [10, .98, .98]))
    parameters88 = result88.x.copy()
    parameters88[0] *= sign88
    kernel_models88[drug88] = two_pole_kernel88(parameters88, 24)
    print("Kernel fit", drug88, "parameters", parameters88.round(5),
          "first 12", kernel_models88[drug88][:12].round(4),
          "six-lag RMSE", np.sqrt(np.mean(result88.fun**2)))
    for factor88, edit88 in [(2.0, "insulin_x2" if sign88 < 0 else "ne_x2"),
                            (.5, "insulin_half" if sign88 < 0 else "ne_half")]:
        rows88 = base_roll.loc[base_roll.edit.eq(edit88), ["stay_id", "hour"]].copy()
        effects88 = []
        for sid88, hour88 in rows88.itertuples(index=False):
            hour88 = int(round(hour88))
            rates88 = tpan.loc[sid88, drug88].fillna(0)
            effect88 = sum(
                (factor88 - 1) * float(rates88.loc[j]) *
                kernel_models88[drug88][hour88 - j - 1]
                for j in range(12, hour88) if j in rates88.index)
            effects88.append(effect88)
        rows88["effect"] = effects88
        by_stay88 = rows88.groupby("stay_id").effect.mean()
        original88 = base_roll.loc[base_roll.edit.eq(edit88)].groupby("stay_id").shift.mean()
        target_rows88.append({"edit": edit88, "stays": len(by_stay88),
                              "kernel_target": by_stay88.mean(),
                              "original_prediction": original88.mean()})
target_table88 = pd.DataFrame(target_rows88)
print("\nPROVISIONAL SUSTAINED TARGETS\n", target_table88.round(4).to_string(index=False))
print("\nHEAD FIXTURE NAMES")
for name88, value88 in list(globals().items()):
    if "74" in name88 and not callable(value88):
        if isinstance(value88, dict):
            print(name88, "dict", list(value88)[:6])
        elif isinstance(value88, pd.DataFrame):
            print(name88, value88.shape, list(value88.columns))
        elif hasattr(value88, "shape"):
            print(name88, tuple(value88.shape))
for key88, value88 in list(pair80.items())[:1]:
    print("pair80", key88, [(type(v).__name__, getattr(v, "shape", None)) for v in value88])
note("Run 88: tested linear action gates including controlled half/double inputs. Fitted a two-pole statistical response model to six record-derived lags to form provisional sustained readout targets; these require longer-lag validation. No weights saved.")
```
