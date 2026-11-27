The state request succeeded, so we now have separate training and tuning representations. I’ll align each reading with the state immediately before it and reconstruct the reading head. If the reconstruction closely matches the forward tool’s factual means, it will provide a reliable basis for computing a small head edit without gradients.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)
print("=== HEAD-RELATED TENSORS ===")
for n in NAMES:
    if n.startswith(("gap", "z_prior", "z_head", "last_", "query_norm",
                     "type_emb", "embed_transition.value", "attr_heads.value",
                     "core.flow", "core.out_norm")):
        print(n, SHAPES[n])
print("numeric_note:", VOCAB.get("numeric_note"))
print("atom_kinds:", VOCAB.get("atom_kinds"))
print("baseline effects:",
      dose_baseline.loc[dose_baseline.reached].groupby("edit")["shift"].mean().to_dict())
print("baseline NLL:", factual_baseline["nll"].mean())

code_ids = {d["name"]: d["id"] for d in VOCAB["codes"]}
numeric_meta = {d["code"]: d for d in VOCAB["numeric"] if d["attribute"] == "value"}
value_centres = VOCAB["value_bin_centres"]

def build_head_data(npz_path, records):
    z = np.load(npz_path)
    obs_codes = set(numeric_meta)
    result = []
    state_rows = []
    for stay in np.unique(z["stay_id"]):
        inds = np.flatnonzero(z["stay_id"] == stay)
        ht = z["hour"][inds].astype(float)
        rec = records.loc[records.stay_id == stay].sort_values("hour", kind="stable")
        last = {}
        for row in rec.itertuples(index=False):
            if row.event not in obs_codes:
                continue
            j = np.searchsorted(ht, float(row.hour) - 1e-5, side="right") - 1
            prev = last.get(row.event, (np.nan, np.nan))
            if j >= 0:
                result.append({
                    "stay_id": int(stay), "hour": float(row.hour), "event": row.event,
                    "observed": float(row.value), "dt": (float(row.hour) - ht[j]) * 3600,
                    "last_value": prev[0],
                    "age": (float(row.hour) - prev[1]) * 3600 if np.isfinite(prev[1]) else 0,
                    "source_hour": ht[j],
                })
                state_rows.append(z["x"][inds[j]])
            last[row.event] = (float(row.value), float(row.hour))
    return pd.DataFrame(result), np.asarray(state_rows, dtype=np.float32)

head_train, head_x_train = build_head_data("/work/fwd_4/states_original.npz", train)
head_tune, head_x_tune = build_head_data("/work/fwd_3/states_original.npz", tuning)
print("\nHead datasets:", head_train.shape, head_tune.shape)
print(head_tune.head(5).to_string(index=False))
print("State-hour sample:",
      np.load("/work/fwd_4/states_original.npz")["hour"][:20].tolist())

with open("/work/notes.md", "a") as f:
    f.write("\n## Head reconstruction setup; Python 17, forward 4\n"
            "- Original states collected for 100 training stays; 100 tuning stays already available.\n"
            "- Constructed reading/head samples using strictly preceding bundle states and same-code historical registers.\n"
            "- No weights changed. Next validate a local head reconstruction against original factual predictions.\n")
```
