The one-step change is concentrated in the intended private channels; leakage into the rest of the state is small. That supports a full-sequence audit, but it does not yet establish factual safety or correct dose effects. I’ll recover the exact stay IDs so the next forward pass can compare factual predictions on the same tuning stays as before.

```python
import pandas as pd

tuning161 = pd.read_csv("/data/tuning.csv")
tuning_ids161 = sorted(tuning161["stay_id"].unique().tolist())[:100]
train_state_ids161 = sorted(np.unique(np.asarray(sid63)).tolist())

print("=== TUNING STAYS FOR PAIRED FACTUAL AUDIT ===")
print(tuning_ids161)
print("\n=== TRAIN STAYS WITH CACHED ORIGINAL STATES ===")
print(train_state_ids161)
print("\n=== EXISTING FORWARD FILES ===")
for directory in sorted(Path("/work").glob("fwd_*")):
    print(directory.name, [p.name for p in directory.iterdir()])

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
```
