The records contain paired sensors, which offer a useful check on treatment-selection bias: disagreement between two measurements of the same quantity may isolate measurement-driven dose variation. I will test this with future readings and a pre-treatment placebo. A strong dose association, no placebo association, and consistent future effects would support—but not prove—the instrument’s validity.

```python
import scipy.linalg as sla

print("=== PANEL LAYOUT ===")
print(panel.index.names, panel.index[:4].tolist())
print(panel.head(4).to_string())
print("Kalman fits:", kalman_fits)

MS = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
DS = ["insulin", "norepinephrine"]
pids = panel.index.get_level_values(0).unique().to_numpy()
phours = panel.index.get_level_values(1).unique().to_numpy()
PN, PT = len(pids), len(phours)
PO = panel[MS].to_numpy(dtype=float).reshape(PN, PT, 4)
PA = panel[DS].fillna(0).to_numpy(dtype=float).reshape(PN, PT, 2)
PF = panel[MS].groupby(level=0).ffill().to_numpy(dtype=float).reshape(PN, PT, 4)
pmed = np.nanmedian(PO.reshape(-1, 4), axis=0)
PF = np.where(np.isfinite(PF), PF, pmed)
print("grid:", PN, PT, phours)
for j, d in enumerate(DS):
    vals, counts = np.unique(PA[:, :, j], return_counts=True)
    print(d, "rates/counts:", list(zip(vals[:25], counts[:25])))

def iv_ratio(d, a, y, groups, C):
    dr = d - C @ sla.lstsq(C, d, lapack_driver="gelsy")[0]
    ar = a - C @ sla.lstsq(C, a, lapack_driver="gelsy")[0]
    yr = y - C @ sla.lstsq(C, y, lapack_driver="gelsy")[0]
    den = np.dot(dr, ar)
    beta_iv = np.dot(dr, yr) / den
    scores = np.bincount(groups, weights=dr * (yr - beta_iv * ar))
    se_iv = np.sqrt(np.dot(scores, scores)) / abs(den)
    slope_a = den / np.dot(dr, dr)
    slope_y = np.dot(dr, yr) / np.dot(dr, dr)
    return beta_iv, se_iv, slope_a, slope_y, dr.std()

iv_rows = []
for dj, (s1, s2) in enumerate(((0, 1), (2, 3))):
    pair = np.isfinite(PO[:, :, s1]) & np.isfinite(PO[:, :, s2])
    delta = PO[:, :, s1] - PO[:, :, s2]
    print("\nPAIR", MS[s1], MS[s2], "n =", pair.sum(),
          "difference mean/sd =", np.nanmean(delta), np.nanstd(delta))
    for lag in (-1, 1, 2, 4):
        for outcome_j in (s1, s2):
            ts = np.arange(2, PT - max(lag, 0))
            yy = PO[:, ts + lag, outcome_j]
            valid = pair[:, ts] & np.isfinite(yy)
            si, ti = np.where(valid)
            tt = ts[ti]
            d = delta[si, tt]
            aa = PA[si, tt, dj]
            past = (PF[si, tt - 1] - pmed) / np.array([40, 40, 12, 12])
            trend = (PF[si, tt - 1] - PF[si, tt - 2]) / np.array([20, 20, 6, 6])
            C_iv = np.column_stack([
                np.ones(len(si)), phours[tt] / 24,
                past, past**2, trend, PA[si, tt - 1]
            ])
            bb, ss, fs, rf, dsd = iv_ratio(d, aa, yy[valid], si, C_iv)
            iv_rows.append(dict(drug=DS[dj], sensor=MS[outcome_j],
                                horizon=lag, n=len(si), beta=bb, se=ss,
                                dose_slope=fs, outcome_slope=rf, instrument_sd=dsd))
iv_table = pd.DataFrame(iv_rows)
print("\n=== SENSOR-DIFFERENCE IV; PAST-ONLY CONTROLS ===")
print(iv_table.round(5).to_string(index=False))
iv_table.to_csv("/work/sensor_iv.csv", index=False)

print("\n=== NUMERIC VOCAB KEYS ===")
for key, val in VOCAB["numeric"].items():
    print(key, str(val)[:1100])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart inspection\n")
    f.write("- Re-read architecture; shared action-state edits can affect every head.\n")
    f.write("- Testing paired-sensor disagreement as a possible dose instrument, with pre-treatment placebo.\n")
    f.write(iv_table.to_string(index=False) + "\n")
    f.write("- Python runs: 8. Forward passes: 3. No edits yet.\n")
```
