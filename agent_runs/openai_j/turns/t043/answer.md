The value head is a two-layer MLP, and we have states for 100 training stays. I’ll test held-out decoding of recent dose innovations from its hidden features, with each drug restricted to its relevant sensors. Reliable decoding would support a small value-head edit; weak decoding would mean we need another approach.

```python
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path

torch.set_num_threads(8)
head_recovery = reconstruct_reading_head(head_meta)
head_hidden = head_recovery["hidden"].detach().cpu().numpy().astype(np.float64)
head_probs = head_recovery["probs"].detach().cpu().numpy()
hm = head_meta.reset_index(drop=True).copy()

pid_index = pd.Index(pids).get_indexer(hm["stay_id"])
reading_hour = np.rint(hm["hour"].to_numpy()).astype(int)
eligible = (pid_index >= 0) & (reading_hour >= 1) & (reading_hour <= 24)
innovation_panel = dose_resid.reshape(len(pids), 24, 2)
recent_innovation = np.zeros((len(hm), 2))
recent_dose = np.zeros((len(hm), 2))
recent_innovation[eligible] = innovation_panel[
    pid_index[eligible], reading_hour[eligible] - 1]
recent_dose[eligible] = PA[
    pid_index[eligible], reading_hour[eligible], :]

gates = np.column_stack([
    hm["event"].isin(["glucose_finger", "glucose_lab"]),
    hm["event"].isin(["map_arterial", "map_cuff"])
]).astype(float)
innovation_targets = recent_innovation * gates
raw_dose_targets = recent_dose * gates

stay_fold = (hm["stay_id"].to_numpy() % 2).astype(int)
decode_results = []
decode_predictions = {}
for target_name, target_array in [
    ("policy_innovation", innovation_targets),
    ("raw_dose", raw_dose_targets)
]:
    for alpha0 in [0.01, 1.0, 10.0, 100.0, 1000.0]:
        prediction0 = np.zeros_like(target_array)
        for fold0 in [0, 1]:
            tr0 = stay_fold != fold0
            te0 = ~tr0
            scaler0 = StandardScaler().fit(head_hidden[tr0])
            regression0 = Ridge(alpha=alpha0).fit(
                scaler0.transform(head_hidden[tr0]), target_array[tr0])
            prediction0[te0] = regression0.predict(
                scaler0.transform(head_hidden[te0]))
        decode_predictions[(target_name, alpha0)] = prediction0
        for dj0, drug0 in enumerate(DRUGS):
            active0 = (gates[:, dj0] > 0) & eligible
            off0 = gates[:, dj0] == 0
            target0 = target_array[active0, dj0]
            pred0 = prediction0[active0, dj0]
            decode_results.append({
                "target": target_name, "alpha": alpha0, "drug": drug0,
                "n": int(active0.sum()),
                "r2_relevant": r2_score(target0, pred0),
                "target_sd": float(target0.std()),
                "prediction_sd": float(pred0.std()),
                "off_sensor_rms": float(np.sqrt(
                    np.mean(prediction0[off0, dj0] ** 2)))
            })
decode_table = pd.DataFrame(decode_results)
print("=== HELD-OUT DOSE DECODING ===")
print(decode_table.to_string(index=False))

print("\n=== RECONSTRUCTED HEAD CHECK ===")
print("hidden shape:", head_hidden.shape)
for e0 in MEAS:
    m0 = hm["event"].eq(e0).to_numpy()
    p0 = head_probs[m0]
    bins0 = hm.loc[m0, "target_bin"].to_numpy(int)
    reconstructed_nll0 = -np.log(np.maximum(
        p0[np.arange(len(p0)), bins0], 1e-12))
    print(e0, "n", int(m0.sum()),
          "original NLL", hm.loc[m0, "original_nll"].mean(),
          "reconstructed NLL", reconstructed_nll0.mean())

print("\n=== BIN CENTRE STRUCTURE ===")
centres0 = VOCAB["value_bin_centres"]
print(type(centres0).__name__)
if isinstance(centres0, dict):
    print({k: str(v)[:180] for k, v in centres0.items()})
else:
    print(str(centres0)[:900])

print("\n=== RESPONSE KERNEL PARAMETERS ===")
for k0, v0 in response_kernels.items():
    if isinstance(v0, (tuple, list)):
        print(k0, [(type(x0).__name__,
                    x0 if np.isscalar(x0) else getattr(x0, "shape", None))
                   for x0 in v0])
    else:
        print(k0, str(v0)[:300])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 37: dose-feature feasibility\n"
            "- Reconstructed deterministic hidden features for all 4,510 "
            "training readings; tested two-fold-by-stay ridge decoding of "
            "sensor-gated dose-policy innovations and raw preceding doses.\n"
            + decode_table.to_string(index=False) + "\n"
            "- No edits yet. Five forward passes used.\n")
```
