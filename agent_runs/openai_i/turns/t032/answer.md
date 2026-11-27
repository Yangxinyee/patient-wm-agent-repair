Insulin’s estimated effect agrees across history adjustment, stay effects, and dose-threshold comparisons. Norepinephrine’s estimate is less independently corroborated, especially at startup. The inspection stopped on a vocabulary-format error; I’ll fix that and test whether dose innovations explain the model’s reading errors and are recoverable from its head features.

```python
import numpy as np
import pandas as pd
import dis
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

print("=== NUMERIC ENTRIES ===")
num27 = VOCAB["numeric"]
entries27 = num27 if isinstance(num27, list) else list(num27.values())
for spec27 in entries27:
    if isinstance(spec27, dict):
        print({k: v for k, v in spec27.items()
               if not isinstance(v, (dict, list))})

print("\n=== HELPER INTERFACES ===")
for nm27 in ("head_exact", "reconstruct_head", "clustered_ols"):
    fn27 = globals().get(nm27)
    if callable(fn27):
        print(nm27, "args:", fn27.__code__.co_varnames[:fn27.__code__.co_argcount],
              "defaults:", fn27.__defaults__)

print("\n=== HEAD_EXACT IMPLEMENTATION ===")
dis.dis(head_exact)

print("\n=== ORIGINAL ERROR VERSUS DOSE INNOVATION ===")
lagkeys27 = pd.MultiIndex.from_arrays(
    [head_rows.stay_id.to_numpy(),
     np.rint(head_rows.hour.to_numpy()).astype(int) - 1],
    names=panel.index.names
)
lagidx27 = panel.index.get_indexer(lagkeys27)
valid27 = (lagidx27 >= 0) & (head_rows.hour.to_numpy() >= 2)
groups27 = head_rows.stay_id.to_numpy()
unique27 = np.unique(groups27)
rng27 = np.random.default_rng(2701)
rng27.shuffle(unique27)
fit_ids27 = unique27[:len(unique27) // 2]
fitmask27 = np.isin(groups27, fit_ids27)
features27 = np.asarray(hidden23, dtype=np.float64)
diagnostics27 = []
decoders27 = {}

for drug27, sensors27 in (
    ("insulin", ["glucose_finger", "glucose_lab"]),
    ("norepinephrine", ["map_arterial", "map_cuff"])
):
    innov27 = np.full(len(head_rows), np.nan)
    dose27 = np.full(len(head_rows), np.nan)
    innov27[valid27] = np.asarray(dose_residuals[drug27])[lagidx27[valid27]]
    dose27[valid27] = panel[drug27].to_numpy()[lagidx27[valid27]]

    for sensor27 in sensors27:
        mask27 = (valid27 & (head_rows.event.to_numpy() == sensor27)
                  & np.isfinite(innov27) & (dose27 > 0))
        xx27 = innov27[mask27]
        yy27 = (head_rows.observed - head_rows.predicted_mean).to_numpy()[mask27]
        xx27 = xx27 - xx27.mean()
        yy27 = yy27 - yy27.mean()
        beta27 = np.dot(xx27, yy27) / np.dot(xx27, xx27)
        scores27 = pd.Series(
            xx27 * (yy27 - beta27 * xx27), index=groups27[mask27]
        ).groupby(level=0).sum().to_numpy()
        se27 = np.sqrt(np.dot(scores27, scores27)) / np.dot(xx27, xx27)
        diagnostics27.append({
            "sensor": sensor27, "n": int(mask27.sum()),
            "missing_effect_slope": beta27, "stay_cluster_se": se27
        })

    eligible27 = (valid27 & head_rows.event.isin(sensors27).to_numpy()
                  & np.isfinite(innov27) & (dose27 > 0))
    tr27 = eligible27 & fitmask27
    te27 = eligible27 & ~fitmask27
    scaler27 = StandardScaler().fit(features27[tr27])
    Xtr27 = scaler27.transform(features27[tr27])
    Xte27 = scaler27.transform(features27[te27])
    print("\nDecoder:", drug27, "train/test readings:", tr27.sum(), te27.sum(),
          "innovation SD:", np.std(innov27[te27]))
    for alpha27 in (10., 100., 1000., 10000.):
        decoder27 = Ridge(alpha=alpha27).fit(Xtr27, innov27[tr27])
        pred27 = decoder27.predict(Xte27)
        mse27 = np.mean((pred27 - innov27[te27]) ** 2)
        r227 = 1 - mse27 / np.var(innov27[te27])
        print(" alpha:", alpha27, "held-out R2:", round(r227, 4),
              "correlation:", round(np.corrcoef(pred27, innov27[te27])[0, 1], 4))
        decoders27[(drug27, alpha27)] = (scaler27, decoder27)

diagnostics27 = pd.DataFrame(diagnostics27)
print("\n", diagnostics27.to_string(index=False))

print("\n=== AVAILABLE STAY SUBSETS ===")
for nm27 in ("train_ids", "tune_ids", "tune_dev", "tune_check", "tune_holdout"):
    obj27 = globals().get(nm27)
    if obj27 is not None:
        arr27 = np.asarray(obj27)
        print(nm27, "shape:", arr27.shape, "first:", arr27.reshape(-1)[:6])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 27; forward 4\n")
    f.write("- No weights changed. Tested original reading errors against dose innovations and split-stay decoding from head features.\n")
    f.write(diagnostics27.to_string(index=False) + "\n")
```