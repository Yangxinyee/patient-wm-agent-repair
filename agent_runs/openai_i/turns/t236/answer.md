The factual result still supports the attention-preserving neutral candidate, but it has no added drug response yet. I’ll inspect the cached replay and readout contracts so I can calibrate a dose-sensitive direction. It must change intervention responses without exhausting the factual-NLL margin.

```python
import pandas as pd
import dis

for name in NAMES:
    W[name] = torch.as_tensor(load_original(name), dtype=torch.float32).clone()
for name, value in neutral209.items():
    W[name] = torch.as_tensor(value, dtype=torch.float32).clone()

fo210 = pd.read_csv("/work/fwd_19/factual_original.csv")
fe210 = pd.read_csv("/work/fwd_19/factual_edited.csv")
print("FACTUAL COLUMNS:", list(fo210.columns))
for col in fo210.columns:
    if ("nll" in col.lower() or "mean" in col.lower()) and pd.api.types.is_numeric_dtype(fo210[col]):
        d210 = fe210[col].to_numpy() - fo210[col].to_numpy()
        print(col, "mean difference", np.nanmean(d210),
              "max absolute difference", np.nanmax(np.abs(d210)))

print("\nREPLAY/READOUT OBJECTS:")
for k, v in list(globals().items()):
    if any(s in k for s in ["182", "183", "191", "192", "193"]):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", len(v), "keys", list(v)[:5])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v))
        elif isinstance(v, (int, float, str)):
            print(k, str(v)[:90])

print("\nREPLAY CONTRACT:")
print("locals:", batch_replay182.__code__.co_varnames)
print(" | ".join(i.opname + (":" + i.argrepr if i.argrepr else "")
                 for i in dis.get_instructions(batch_replay182)))
print("\nREADOUT CONTRACT:")
print("batch_q182 locals:", batch_q182.__code__.co_varnames)
print("capture159 locals:", capture159.__code__.co_varnames)
print("c1 fields:")
for k, v in c1.items():
    print(k, tuple(v.shape) if hasattr(v, "shape") else str(v)[:100])

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 18; Python 210 — attention-preserving neutral passes local safeguard\n"
            "- Original and neutral209 tuning factual NLL both 1.63155 on 100 stays; per-event scores unchanged at displayed precision.\n"
            "- Six latent-path tensors edited, no attention changes, zero innovation-response columns.\n"
            "- Next: controlled latent-token response measurements, closed-form calibration, then GPU checks.\n")
print("ACTIVE EDITS:", edited())
```