The prediction errors point in the expected directions, but dose-innovation decoding from the value-head features failed on held-out stays. That weakens the proposed head-only approach. I’ll check event alignment and the raw recurrent state: recoverable dose information there would support changing the head’s input path; failure there would require an action-path edit.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

print("=== EVENT TIMING, FIRST TRAINING STAY ===")
raw28 = pd.read_csv("/data/train.csv", nrows=150)
sid28 = raw28.stay_id.iloc[0]
print(raw28.loc[raw28.stay_id == sid28,
                ["hour", "event", "value", "rate", "dose"]].head(45).to_string(index=False))

print("\n=== STATE ARCHIVE FIELDS ===")
with np.load("/work/fwd_3/states_original.npz") as pack28:
    for key28 in pack28.files:
        arr28 = pack28[key28]
        print(key28, arr28.shape, arr28.dtype,
              arr28.reshape(-1)[:5] if arr28.ndim < 2 else "")

print("\n=== FIRST READING ALIGNMENTS ===")
print(head_rows[["stay_id", "hour", "event", "previous_state"]].head(18).to_string(index=False))
print("state_hours:", np.asarray(state_hours)[:18])
print("previous_state range:",
      head_rows.previous_state.min(), head_rows.previous_state.max())

state_read28 = np.asarray(state_x)[head_rows.previous_state.to_numpy(dtype=int)]
feature_sets28 = {
    "head_hidden": np.asarray(hidden23),
    "out": state_read28[:, :192],
    "mean_h": state_read28[:, 192:960].reshape(-1, 4, 192).mean(axis=1),
    "all_h": state_read28[:, 192:960],
    "posterior_z": state_read28[:, 960:992],
}
decode_table28 = []

for drug28, sensors28 in (
    ("insulin", ["glucose_finger", "glucose_lab"]),
    ("norepinephrine", ["map_arterial", "map_cuff"])
):
    dose28 = np.full(len(head_rows), np.nan)
    innovation28 = np.full(len(head_rows), np.nan)
    dose28[valid27] = panel[drug28].to_numpy()[lagidx27[valid27]]
    innovation28[valid27] = np.asarray(dose_residuals[drug28])[lagidx27[valid27]]
    eligible28 = (valid27 & head_rows.event.isin(sensors28).to_numpy()
                  & np.isfinite(innovation28) & (dose28 > 0))
    tr28 = eligible28 & fitmask27
    te28 = eligible28 & ~fitmask27
    for feature_name28, feature28 in feature_sets28.items():
        scaler28 = StandardScaler().fit(feature28[tr28])
        Xtr28 = scaler28.transform(feature28[tr28])
        Xte28 = scaler28.transform(feature28[te28])
        for target_name28, target28 in (
            ("actual_rate", dose28), ("dose_innovation", innovation28)
        ):
            best28 = None
            for alpha28 in (10., 100., 1000., 10000.):
                reg28 = Ridge(alpha=alpha28).fit(Xtr28, target28[tr28])
                pred28 = reg28.predict(Xte28)
                r228 = 1 - np.mean((pred28 - target28[te28]) ** 2) / np.var(target28[te28])
                row28 = {
                    "drug": drug28, "features": feature_name28,
                    "target": target_name28, "alpha": alpha28,
                    "heldout_R2": r228,
                    "correlation": np.corrcoef(pred28, target28[te28])[0, 1]
                }
                if best28 is None or r228 > best28["heldout_R2"]:
                    best28 = row28
            decode_table28.append(best28)

decode_table28 = pd.DataFrame(decode_table28)
print("\n=== SPLIT-STAY DECODING DIAGNOSTIC ===")
print(decode_table28.round(4).to_string(index=False))

print("\n=== TUNING PANEL / ROLLOUT FIELDS ===")
print("tpanel:", type(tpanel).__name__, getattr(tpanel, "shape", None),
      getattr(getattr(tpanel, "index", None), "names", None))
dose_file28 = pd.read_csv("/work/fwd_2/dose_edit_original.csv")
print("rollout columns:", list(dose_file28.columns))
print(dose_file28.head(3).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 28; forward 4\n")
    f.write("- Head-feature dose-innovation decoder failed on 50 held-out training stays; not safe to edit from it alone.\n")
    f.write("- Checked event timing and compared raw-state dose decodability.\n")
    f.write(decode_table28.round(4).to_string(index=False) + "\n")
```