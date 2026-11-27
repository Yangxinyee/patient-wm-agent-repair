The kernel ran out of memory before reporting results; no weights had been edited. I’ll rebuild only the compact hourly panel and the records needed for the 200 state-sampled stays, using chunked reads. Stable memory use and successful state alignment will let us resume; another failure would require smaller batches.

```python
import gc, json, resource
from pathlib import Path
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)
VOCAB = json.loads(Path("/model/vocab.json").read_text())
print("Weight helpers available:", all(k in globals() for k in
      ["load", "load_original", "save_edit", "drop_edit", "NAMES", "SHAPES"]))

events = ["glucose_finger", "glucose_lab", "insulin",
          "map_arterial", "map_cuff", "norepinephrine"]
event_col = {s: j for j, s in enumerate(events)}
base_sid = 1010000000
train_ids = np.arange(base_sid, base_sid + 30000, dtype=np.int64)
hour_grid = np.arange(-1, 25)
pa = np.full((30000 * 26, 6), np.nan, dtype=np.float32)
pa[:, [2, 5]] = 0
selected_chunks = []
dtypes = {"stay_id": "int64", "hour": "float32", "value": "float32", "rate": "float32"}
for chunk in pd.read_csv("/data/train.csv",
                         usecols=["stay_id", "hour", "event", "value", "rate"],
                         dtype=dtypes, chunksize=100000):
    selected_chunks.append(chunk.loc[chunk.stay_id < base_sid + 100].copy())
    for ev, col in event_col.items():
        part = chunk.loc[chunk.event == ev]
        hh = np.floor(part.hour.to_numpy() + 1e-4).astype(np.int64)
        st = part.stay_id.to_numpy() - base_sid
        valid = (st >= 0) & (st < 30000) & (hh >= -1) & (hh <= 24)
        loc = st[valid] * 26 + hh[valid] + 1
        vals = part["rate" if ev in ("insulin", "norepinephrine") else "value"].to_numpy()
        pa[loc, col] = vals[valid]
    del chunk, part
train_small = pd.concat(selected_chunks, ignore_index=True)
del selected_chunks
panel = pd.DataFrame(pa, index=pd.MultiIndex.from_product(
    [train_ids, hour_grid], names=["stay_id", "t"]), columns=events)
tuning = pd.read_csv("/data/tuning.csv",
                     usecols=["stay_id", "hour", "event", "value", "rate"], dtype=dtypes)

def compact_panel(df):
    stays = np.sort(df.stay_id.unique())
    out = pd.DataFrame(index=pd.MultiIndex.from_product(
        [stays, hour_grid], names=["stay_id", "t"]), columns=events, dtype=np.float32)
    out.loc[:, ["insulin", "norepinephrine"]] = 0
    for ev in events:
        q = df.loc[df.event == ev]
        ix = pd.MultiIndex.from_arrays(
            [q.stay_id.to_numpy(), np.floor(q.hour.to_numpy() + 1e-4).astype(int)])
        out.loc[ix, ev] = q["rate" if ev in ("insulin", "norepinephrine") else "value"].to_numpy()
    return out

tpanel = compact_panel(tuning)
factual_baseline = pd.read_csv("/work/fwd_2/factual_original.csv")
dose_baseline = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print("Panel / selected records:", panel.shape, train_small.shape, tuning.shape)
print("Baseline NLL:", factual_baseline.nll.mean())
print("Baseline shifts:",
      dose_baseline.loc[dose_baseline.reached].groupby("edit")["shift"].mean().to_dict())
print("Peak resident memory (platform units):", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print("\nHead-related weights:")
for n in NAMES:
    if n.startswith(("gap", "z_prior", "z_head", "last_", "query_norm", "type_emb",
                     "embed_transition.value", "attr_heads.value", "core.flow", "core.out_norm")):
        print(n, SHAPES[n])
print("numeric_note:", VOCAB.get("numeric_note"))
print("atom_kinds:", VOCAB.get("atom_kinds"))
gc.collect()
with open("/work/notes.md", "a") as f:
    f.write("\n## Kernel recovery; Python 18, forward 4\n"
            "- Python 17 exceeded memory before producing results. No edits existed.\n"
            "- Rebuilt a float32 hourly panel by chunked CSV reads; retained raw training records only for 100 state-sampled stays.\n"
            "- Continuing with bounded-memory head reconstruction and validation.\n")
```
