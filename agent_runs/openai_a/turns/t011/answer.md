Sensor disagreement could help identify insulin’s effect, but using the same noisy reading in the outcome change can create regression-to-the-mean bias. I’ll test future reading levels directly and check preceding readings as placebos. A strong dose association, no pre-treatment association, and consistent future responses would support this instrument.

```python
import inspect
import scipy.linalg as sla

print("=== PANEL AND EXISTING ESTIMATOR ===")
print("index:", panel.index.names, "hours:", panel.index.get_level_values(-1).unique().tolist())
print(panel.head(5).round(3).to_string())
print("tuning stay ids:", tpanel.index.get_level_values(0).unique()[:12].tolist())
try:
    print(inspect.getsource(sensor_iv))
except Exception:
    print("sensor_iv names:", sensor_iv.__code__.co_names)
    print("sensor_iv constants:", sensor_iv.__code__.co_consts)

sid = panel.index.get_level_values(0).to_numpy()
hours = panel.index.get_level_values(-1).to_numpy(dtype=float)
pg = panel.groupby(level=0, sort=False)

# These controls precede the instrument and the dose; no current sensor enters.
control_cols = [np.ones(len(panel)), hours / 24]
for col in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
            "insulin", "norepinephrine"]:
    for lag in [1, 2]:
        arr = pg[col].shift(lag).to_numpy()
        finite = np.isfinite(arr)
        fill = np.nanmedian(arr)
        sd = np.nanstd(arr)
        control_cols += [np.where(finite, (arr-fill)/max(sd, 1e-6), 0), (~finite).astype(float)]
preC = np.column_stack(control_cols)

def contrast_iv(drug, sensor1, sensor2, outcome, horizon, adjusted=True):
    z0 = (panel[sensor1] - panel[sensor2]).to_numpy()
    u0 = panel[drug].fillna(0).to_numpy()
    y0 = pg[outcome].shift(-horizon).to_numpy()
    good = np.isfinite(z0) & np.isfinite(y0)
    z, u, y = z0[good], u0[good], y0[good]
    C = preC[good] if adjusted else np.ones((good.sum(), 1))
    vals = np.column_stack([z, u, y])
    resid = vals - C @ sla.lstsq(C, vals, lapack_driver="gelsy")[0]
    z, u, y = resid.T
    zu = z @ u
    b = (z @ y) / zu
    groups, inv = np.unique(sid[good], return_inverse=True)
    gm = np.bincount(inv, weights=z*(y-b*u), minlength=len(groups))
    se = np.sqrt(gm @ gm) / abs(zu)
    fs = zu / (z @ z)
    fg = np.bincount(inv, weights=z*(u-fs*z), minlength=len(groups))
    fsse = np.sqrt(fg @ fg)/(z @ z)
    return dict(n=int(good.sum()), beta=float(b), se=float(se),
                first_stage=float(fs), first_stage_t=float(fs/fsse))

contrast_results = {}
print("\n=== SENSOR-CONTRAST IV: FUTURE LEVELS AND PRE-TREATMENT PLACEBOS ===")
for drug, s1, s2 in [
    ("insulin", "glucose_finger", "glucose_lab"),
    ("norepinephrine", "map_arterial", "map_cuff")
]:
    for adj in [False, True]:
        for k in [-2, -1, 1, 2, 3, 4, 6, 8]:
            r = contrast_iv(drug, s1, s2, s1, k, adj)
            contrast_results[(drug, k, adj)] = r
            print(drug, "horizon", k, "prior_controls", adj,
                  "b", round(r["beta"], 4), "SE", round(r["se"], 4),
                  "first-stage t", round(r["first_stage_t"], 2), "n", r["n"])

with open("/work/notes.md", "a") as f:
    f.write("\n## Sensor-contrast local projections; Python 9, forward 2\n"
            "- Reimplemented IV on future reading levels, controlling only pre-treatment history.\n"
            "- Pre-treatment placebo horizons test independence; weak first stages remain disqualifying.\n"
            + json.dumps({str(k): v for k, v in contrast_results.items()}, indent=2) + "\n")
```