This repeats the recovery output, so no new experiment has run and no weights have changed. I’ll now inspect the saved effect estimates and calculate their implied rollout shifts. Agreement between the two glucose sensors and between alternative estimators would strengthen the targets; disagreement would make them provisional.

```python
import pandas as pd
import json

print("TRAIN IDS:", train_ids[:5], "TUNING IDS:", tuning_ids[:5])
print("\nNUMERIC TABLES:")
for entry in VOCAB["numeric"]:
    print(entry)
print("\nOTHER EMBEDDING TENSORS:")
for name in NAMES:
    if name.startswith(("embed_transition", "embed.kq", "embed.numeric")):
        print(name, SHAPES[name])
print("\nARRAY ALIGNMENT:")
for ev in events6:
    print(ev, A[ev].shape, "first stay:", A[ev][0, :8],
          "tuning average:", np.nanmean(AT[ev]))
print("\nIMPULSE ESTIMATES:")
for key, value in impulse_results.items():
    print(key, value)
print("\nHISTORY FEATURES:")
print(feature_names)
print("\nROLLOUT SUMMARY:")
print(Path("/work/fwd_1/summary.json").read_text())
roll0 = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
fact0 = pd.read_csv("/work/fwd_2/factual_original.csv")
print("ROLLOUT:", roll0.shape, list(roll0.columns))
print(roll0.head(5).to_string(index=False))
print("FACTUAL:", fact0.shape, list(fact0.columns))
print(fact0.head(5).to_string(index=False))

kernel_refs = {}
for (drug, sensor), result in impulse_results.items():
    if isinstance(result, dict):
        kernel = result.get("theta", result.get("kernel", result.get("effects")))
        print("kernel keys:", drug, sensor, list(result))
    else:
        kernel = result
    if kernel is None:
        continue
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 1:
        continue
    D = np.nan_to_num(AT[drug][:, 1:26])
    reference = np.zeros((len(tuning_ids), 12))
    exposed = np.zeros_like(reference, dtype=bool)
    for j, hour in enumerate(range(13, 25)):
        for dose_hour in range(12, hour):
            lag = hour - dose_hour - 1
            if lag < len(kernel):
                reference[:, j] += D[:, dose_hour] * kernel[lag]
        exposed[:, j] = np.any(D[:, 12:hour] > 0, axis=1)
    per_stay = np.divide((reference * exposed).sum(1), exposed.sum(1),
                         out=np.full(len(tuning_ids), np.nan),
                         where=exposed.sum(1) > 0)
    kernel_refs[(drug, sensor)] = (reference, exposed, per_stay)
    print("REFERENCE x2", drug, sensor,
          "all 400:", np.nanmean(per_stay),
          "first 100:", np.nanmean(per_stay[:100]),
          "x1/2:", -0.5 * np.nanmean(per_stay[:100]))

with open("/work/notes.md", "a") as f:
    f.write("\nPython runs: 9; forwards: 2. Recovered architecture and variables. "
            "Original factual value NLL on first 100 tuning stays is 1.63982. "
            "No weights edited. Preparing record-derived rollout targets.\n")
```
