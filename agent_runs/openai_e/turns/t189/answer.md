The training-state request succeeded for both models. It provides calibration inputs, not new evidence about drug effects. I’ll check the reading-register layout and decoder computation so each prediction uses only earlier events. A mismatch with the established helper would halt the calibration attempt.

```python
import dis, numpy as np, pandas as pd, torch
from pathlib import Path

print("=== REGISTER AND DECODER WEIGHTS ===")
for n161 in NAMES:
    if any(s in n161 for s in ["last_numeric","last_status","z_prior","z_head","gap.","query_norm","type_emb"]):
        print(n161, SHAPES[n161])
print("=== NUMERIC VOCAB EXAMPLE ===")
print(repr(VOCAB["numeric"])[:2200])
print("=== TRUSTED STATIC EXAMPLE ===")
print("head metadata:", head_meta.iloc[:3][
    ["code","last_z","last_bin","age","has_last"]].to_dict("records"))
print("register8:", register8[:3].tolist())
print("=== COMPACT READOUT BYTECODE ===")
print("\n".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(readout125)
                if i.opname not in ["RESUME","PRECALL","CACHE"])[:8500])

train_ids161 = set(range(1010000000,1010000100))
train_parts161 = []
for c161 in pd.read_csv("/data/train.csv",
        usecols=["stay_id","hour","event","value","rate"], chunksize=200000):
    train_parts161.append(c161[c161.stay_id.isin(train_ids161)])
train_records161 = pd.concat(train_parts161, ignore_index=True)
s161 = np.load("/work/fwd_22/states_original.npz")
print("=== ALIGNMENT INPUTS ===")
print("Filtered records:", train_records161.shape)
print(train_records161.head(8).to_string(index=False))
print("First state hours:", s161["hour"][:12].tolist())
with open("/work/notes.md","a") as f:
    f.write("\n## Forward 22 / Run 161: training calibration inputs\n"
            "Collected paired original/insulin-baseline states on 100 training stays. "
            "Audited decoder/register computation and filtered exact matching records before calibration. "
            "No weights changed. Python runs: 161; forwards: 22.\n")
```