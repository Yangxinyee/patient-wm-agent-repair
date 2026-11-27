The records contain hourly readings followed about five minutes later by one-hour infusions, with substantial variation in dose. That gives a useful temporal ordering, but dose–outcome correlations could still reflect clinicians responding to the patient. I’ll compare raw associations with history-adjusted and within-stay estimates. Consistent estimates would support a drug effect; disagreement would call for stronger identification checks.

```python
from scipy.linalg import lstsq

print("=== NUMERIC ENCODING ===")
for item in VOCAB["numeric"]:
    if item["code"] in ["insulin", "norepinephrine", "glucose_finger", "map_arterial"]:
        print(item)

print("\n=== EVENT TIMING ===")
for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
              "insulin", "norepinephrine"]:
    a = train.loc[train.event == event, "hour"].to_numpy()
    print(event, "fractional-hour frequencies:",
          pd.Series(np.round(a - np.floor(a), 4)).value_counts().head(4).to_dict())

def make_panel(frame):
    relevant = frame.event.isin([
        "glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
        "insulin", "norepinephrine"])
    a = frame.loc[relevant, ["stay_id", "hour", "event", "value", "rate"]].copy()
    a["t"] = np.floor(a.hour + 1e-4).astype("int16")
    a["x"] = a["value"].fillna(a["rate"])
    wide = a.pivot_table(index=["stay_id", "t"], columns="event", values="x",
                         aggfunc="first", observed=True)
    index = pd.MultiIndex.from_product(
        [np.sort(frame.stay_id.unique()), np.arange(-1, 25)],
        names=["stay_id", "t"])
    wide = wide.reindex(index).sort_index()
    wide.columns.name = None
    wide[["insulin", "norepinephrine"]] = wide[
        ["insulin", "norepinephrine"]].fillna(0)
    return wide

panel = make_panel(train)
tpanel = make_panel(tuning)
pgroup = panel.groupby(level="stay_id", sort=False)
time_array = panel.index.get_level_values("t").to_numpy()
stay_array = panel.index.get_level_values("stay_id").to_numpy()
measurements = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
history = pgroup[measurements].ffill()
history = history.fillna(history.median())
history_features = []
history_names = []
for event in measurements:
    scale = 50.0 if "glucose" in event else 20.0
    centre = history[event].median()
    x = (history[event].to_numpy() - centre) / scale
    history_features.extend([x, x*x,
                             panel[event].notna().to_numpy().astype(float)])
    history_names.extend([event, event+"^2", event+"_observed"])
    for lag in [1, 2]:
        a = history[event].groupby(level="stay_id").shift(lag).fillna(centre)
        history_features.append((a.to_numpy()-centre)/scale)
        history_names.append(event+f"_lag{lag}")
for drug in ["insulin", "norepinephrine"]:
    for lag in [1, 2]:
        history_features.append(
            pgroup[drug].shift(lag).fillna(0).to_numpy())
        history_names.append(drug+f"_lag{lag}")
history_features.extend([time_array/24, (time_array/24)**2])
history_names.extend(["hour", "hour^2"])
HX = np.column_stack(history_features).astype(np.float64)

def estimate_transition(outcome, drug, mode):
    y0 = panel[outcome].to_numpy()
    y1 = pgroup[outcome].shift(-1).to_numpy()
    d = panel[drug].to_numpy().astype(float)
    valid = (time_array >= 0) & (time_array < 24) & np.isfinite(y0) & np.isfinite(y1)
    ix = np.flatnonzero(valid)
    Y = (y1-y0)[ix].astype(float)
    D = d[ix]
    if mode == "raw":
        X = np.ones((len(ix), 1))
    elif mode == "current":
        z = y0[ix].astype(float)
        X = np.column_stack([np.ones(len(ix)), z, z*z, time_array[ix]/24])
    else:
        other = "norepinephrine" if drug == "insulin" else "insulin"
        X = np.column_stack([np.ones(len(ix)), HX[ix],
                             panel[other].to_numpy()[ix]])
    groups, ids = pd.factorize(stay_array[ix], sort=False)
    if mode == "history+stayFE":
        counts = np.bincount(groups)
        for j in range(X.shape[1]):
            means = np.bincount(groups, weights=X[:, j]) / counts
            X[:, j] -= means[groups]
        Y = Y - (np.bincount(groups, weights=Y)/counts)[groups]
        D = D - (np.bincount(groups, weights=D)/counts)[groups]
    bd = lstsq(X, D, lapack_driver="gelsy")[0]
    by = lstsq(X, Y, lapack_driver="gelsy")[0]
    dr = D-X@bd
    yr = Y-X@by
    den = dr@dr
    beta = (dr@yr)/den
    resid = yr-beta*dr
    scores = np.bincount(groups, weights=dr*resid)
    se = np.sqrt(scores@scores)/den
    print(outcome, drug, mode, "n=", len(ix),
          "effect per rate unit=", round(beta, 5),
          "cluster SE=", round(se, 5),
          "residual dose SD=", round(np.std(dr), 5))
    return {"beta": beta, "se": se, "n": len(ix),
            "dose_resid_sd": np.std(dr)}

transition_results = {}
print("\n=== ONE-HOUR CHANGE ESTIMATES ===")
for outcome, drug in [("glucose_finger", "insulin"),
                      ("map_arterial", "norepinephrine")]:
    for mode in ["raw", "current", "history", "history+stayFE"]:
        transition_results[(outcome, drug, mode)] = estimate_transition(outcome, drug, mode)

with open("/work/notes.md", "a") as f:
    f.write("\n## Record transition analysis; Python 3, forward 0\n")
    f.write("- 30,000 training stays, 400 tuning stays; readings hourly and drugs at hour + 0.0833.\n")
    f.write("- Built an hourly panel; missing infusion records mean no recorded infusion.\n")
    f.write("- Compared raw, current-value, history-adjusted and stay-fixed-effect one-hour changes.\n")
    f.write(json.dumps({str(k): v for k, v in transition_results.items()}, indent=2)+"\n")
```