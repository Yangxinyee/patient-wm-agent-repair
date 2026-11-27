Correcting sensor noise raises the norepinephrine estimates to roughly +3 to +4, but that depends on assumptions about dosing and measurement errors. The first lag-based test was too small and imprecise. I’ll repeat it with more usable histories and add admission-state controls. A stable positive estimate across these checks would be much more persuasive.

```python
def admission_corrected(outcome, proxy, drug):
    y = pgroup[outcome].shift(-1).to_numpy().astype(float)
    cur = panel[outcome].to_numpy().astype(float)
    prx = panel[proxy].to_numpy().astype(float)
    d = panel[drug].to_numpy().astype(float)
    initial = panel.xs(0, level="t")
    a0 = pd.Series(stay_array).map(initial[outcome]).to_numpy().astype(float)
    c0 = pd.Series(stay_array).map(initial[proxy]).to_numpy().astype(float)
    good = ((time_array >= 1) & (time_array < 24) &
            np.isfinite(np.column_stack([y, cur, prx, a0, c0])).all(axis=1))
    ix = np.flatnonzero(good)
    X = np.column_stack([np.ones(len(ix)), time_array[ix]/24,
                         (time_array[ix]/24)**2])
    R = np.column_stack([cur[ix], d[ix], a0[ix]])
    Q = np.column_stack([prx[ix], d[ix], c0[ix]])
    b, se, cond = iv_fit(y[ix], R, Q, X, stay_array[ix])
    print(drug, "admission-corrected", "n:", len(ix),
          "a,b,baseline:", np.round(b, 4), "SE:", np.round(se, 4))
    return {"coefficients": b.tolist(), "se": se.tolist(), "n": len(ix)}

def larger_difference_iv(outcome, drug, lags=(2,3), centre=True, proxy=None):
    now = panel[outcome].to_numpy().astype(float)
    prev = pgroup[outcome].shift(1).to_numpy().astype(float)
    future = pgroup[outcome].shift(-1).to_numpy().astype(float)
    dose = panel[drug].to_numpy().astype(float)
    dose_prev = pgroup[drug].shift(1).to_numpy().astype(float)
    base = pd.Series(stay_array).map(panel.xs(0, level="t")[outcome]).to_numpy()
    qlist = []
    for lag in lags:
        q = pgroup[outcome].shift(lag).to_numpy().astype(float)
        qlist.append(q-base if centre else q)
        qlist.append(pgroup[drug].shift(lag).to_numpy().astype(float))
    if proxy is not None:
        qlist.append(pgroup[proxy].shift(1).to_numpy().astype(float))
    Q = np.column_stack(qlist)
    good = ((time_array >= max(lags)+1) & (time_array < 24) &
            np.isfinite(np.column_stack([now, prev, future, Q])).all(axis=1))
    ix = np.flatnonzero(good)
    Y = (future-now)[ix]
    R = np.column_stack([(now-prev)[ix], (dose-dose_prev)[ix]])
    # Hour effects allow a shared nonconstant time trend.
    X = np.eye(25)[time_array[ix].astype(int)]
    b, se, condition = iv_fit(Y, R, Q[ix], X, stay_array[ix])
    print(drug, outcome, "difference IV", "lags:", lags,
          "centred:", centre, "proxy:", proxy, "n:", len(ix),
          "a,b:", np.round(b,4), "SE:", np.round(se,4),
          "condition:", round(condition,1))
    return {"persistence": b[0], "effect": b[1], "se_persistence": se[0],
            "se_effect": se[1], "n": len(ix), "condition": condition}

print("=== ADMISSION-CONTROLLED REPLICATE-SENSOR ESTIMATES ===")
identification_results = {}
for outcome, proxy, drug in [
    ("map_arterial", "map_cuff", "norepinephrine"),
    ("map_cuff", "map_arterial", "norepinephrine")]:
    identification_results[(outcome, "admission")] = admission_corrected(outcome, proxy, drug)

print("\n=== LARGER FIRST-DIFFERENCE IV ESTIMATES ===")
for outcome, drug in [("glucose_finger", "insulin"),
                      ("map_arterial", "norepinephrine"),
                      ("map_cuff", "norepinephrine")]:
    for lags in [(2,3), (2,3,4)]:
        for centre in [False, True]:
            identification_results[(outcome, str(lags), centre)] = larger_difference_iv(
                outcome, drug, lags=lags, centre=centre)

print("\n=== SENSOR ERROR MOMENTS ===")
for a,b in [("map_arterial","map_cuff"), ("glucose_finger","glucose_lab")]:
    pair = panel[[a,b]].dropna()
    cov = pair.cov().to_numpy()
    print(a,b, "replicate-noise variance estimates:",
          np.round([cov[0,0]-cov[0,1], cov[1,1]-cov[0,1]],3))

with open("/work/notes.md", "a") as f:
    f.write("\n## More efficient dynamic checks; Python 6, forward 2\n")
    f.write("- Replicate-sensor NE estimates are positive but rely on dose/error independence.\n")
    f.write("- Insulin replicate-sensor level estimates are invalid if dosing uses finger noise; "
            "the negative sensor-difference IV is more defensible for insulin.\n")
    f.write("- Enlarged first-difference samples using own-sensor instruments at lags >=2, "
            "which avoid overlapping current measurement errors.\n")
    f.write(json.dumps({str(k):v for k,v in identification_results.items()}, indent=2)+"\n")
```