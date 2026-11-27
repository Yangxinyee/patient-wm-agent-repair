The attribute register has eight inputs, while the architecture gives only a shortened description. Its layout is now a concrete suspect. I’ll compare plausible layouts against original GPU predictions without editing weights. A sharp improvement in both NLL and predicted means would support a register-layout diagnosis.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
orig231 = np.load("/work/fwd_15/states_original.npz")
p231, wc231, q231, zp231 = head225(WO225, orig231, noise=None, status=None, typeidx=1)
basewc231 = wc231 - lin225(head_ln, "last_numeric", WO225)

has231 = torch.tensor(head_rows.last_value.notna().to_numpy(), dtype=torch.float32)
age231 = torch.tensor(
    np.log1p(np.maximum(0, (head_rows.hour - head_rows.last_hour).fillna(0).to_numpy()) * 3600) / 10,
    dtype=torch.float32)
z231 = torch.zeros(len(head_rows))
labels231 = np.zeros(len(head_rows), dtype=np.int64)
centres231 = np.zeros((len(head_rows), 17), dtype=np.float32)
for ev in head_rows.event.unique():
    ix = np.flatnonzero(head_rows.event.to_numpy() == ev)
    spec = num_spec[(ev, "value")]
    last = head_rows.last_value.iloc[ix].fillna(spec["median"]).to_numpy()
    z231[ix] = torch.tensor(np.clip((last - spec["median"]) / spec["iqr"], -8, 8), dtype=torch.float32)
    labels231[ix] = np.searchsorted(spec["fine_bin_edges"], head_rows.value.iloc[ix].to_numpy(), side="right") + 1
    centres231[ix, 1:] = np.asarray(VOCAB["value_bin_centres"][ev], dtype=np.float32)

zero231 = torch.zeros_like(has231)
layouts231 = {
    "current": [has231, zero231, zero231, z231, zero231, zero231, age231, zero231],
    "presence_age_numeric6": [has231, age231, z231, has231, zero231, zero231, zero231, zero231],
    "numeric6_age_presence": [z231, has231, zero231, zero231, zero231, zero231, age231, has231],
    "presence_age_values": [has231, age231, z231, zero231, zero231, zero231, zero231, zero231],
    "presence_flags_values_age": [has231, has231, zero231, zero231, z231, zero231, zero231, age231],
    "flags_values_age_presence": [has231, zero231, zero231, z231, zero231, zero231, age231, has231],
    "presence_numeric6_age": [has231, z231, has231, zero231, zero231, zero231, zero231, age231],
    "numeric6_presence_age": [z231, has231, zero231, zero231, zero231, zero231, has231, age231],
}
gpu231 = pd.read_csv("/work/fwd_20/factual_original.csv")
join231 = head_rows[["stay_id", "hour", "event"]].assign(local_ix=np.arange(len(head_rows))).merge(
    gpu231, on=["stay_id", "hour", "event"], how="inner", validate="one_to_one")
ix231 = join231.local_ix.to_numpy()
trans231 = WO225[trname225][codes225 * 17 + torch.tensor(head_vb)]
results231 = []
for lname, fields in layouts231.items():
    ln = torch.stack(fields, dim=1)
    for status in [None, 0, 1]:
        wc = basewc231 + lin225(ln, "last_numeric", WO225)
        if status is not None:
            wc = wc + WO225["last_status.weight"][status]
        logits = lin225(F.gelu(lin225(wc, "attr_heads.value_bin.0", WO225)),
                        "attr_heads.value_bin.2", WO225) + trans231
        probs = torch.softmax(logits, dim=-1).numpy()
        nll = -np.log(np.maximum(probs[np.arange(len(probs)), labels231], 1e-30))
        means = (probs * centres231).sum(1)
        results231.append({"layout": lname, "status": status,
                           "nll": float(nll[ix231].mean()),
                           "gpu_nll": float(join231.nll.mean()),
                           "mean_rmse": float(np.sqrt(np.mean((means[ix231] - join231.predicted_mean.to_numpy())**2)))})
print(pd.DataFrame(results231).sort_values("mean_rmse").to_string(index=False))
print("joined readings", len(join231), "fresh/existing labels differ",
      int(np.sum(labels231 != _labels218)))
```
