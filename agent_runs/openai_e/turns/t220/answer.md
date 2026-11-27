The installed tensor exactly matches the saved insulin baseline. Paired training states are available, and the NE candidates range from rank one to several directions. I’ll reconstruct prelude inputs under the insulin repair and compare them with the original inputs. If the old NE edit leaks more on repaired states, anchoring there could improve its factual-fit trade-off.

```python
import inspect, pandas as pd, numpy as np, torch
from pathlib import Path

print("SAFE HELPER SIGNATURES")
for fn190 in ["recon_record_atom", "recon_sets", "prelude46", "posterior94"]:
    f190 = globals()[fn190]
    params190 = []
    for p190 in inspect.signature(f190).parameters.values():
        d190 = p190.default
        shown190 = (repr(d190) if isinstance(d190, (str, int, float, bool, type(None)))
                    else "<required>" if d190 is inspect.Parameter.empty
                    else f"<{type(d190).__name__}>")
        params190.append((p190.name, shown190))
    print(fn190, params190)

print("\nUNIT MAPS")
for key190 in ["RECON_UNITS", "RECON_RATE_UNITS", "CODE_ID"]:
    print(key190, globals().get(key190))

print("\nRETAINED FULL RECORD TABLES")
full190 = []
for k190, v190 in list(globals().items()):
    if isinstance(v190, pd.DataFrame) and all(
        c in v190.columns for c in ["stay_id", "event", "dose", "rate_unit", "end_hour"]
    ):
        full190.append((k190, len(v190)))
print(full190[:12])

with np.load("/work/fwd_22/states_original.npz") as a190:
    train_ids190 = np.unique(a190["stay_id"])
    train_state190 = {k: a190[k].copy() for k in a190.files}
with np.load("/work/fwd_22/states_edited.npz") as a190:
    base_state190 = {k: a190[k].copy() for k in a190.files}

if full190:
    candidate190 = globals()[full190[0][0]]
    train_full190 = candidate190[candidate190.stay_id.isin(train_ids190)].copy()
else:
    train_full190 = pd.DataFrame()
if train_full190.empty or train_full190.stay_id.nunique() != len(train_ids190):
    train_full190 = pd.concat([
        c190[c190.stay_id.isin(train_ids190)]
        for c190 in pd.read_csv("/data/train.csv", chunksize=200000)
    ], ignore_index=True)

print("\nTRAINING RECORD SAMPLE")
print("stays", len(train_ids190), "records", len(train_full190),
      "columns", list(train_full190.columns))
print(train_full190[train_full190.event.isin(["insulin", "norepinephrine"])]
      .head(3).to_string(index=False))
print("state first rows",
      list(zip(train_state190["stay_id"][:6], train_state190["hour"][:6],
               train_state190["n_atoms"][:6])))

print("\nVERIFIED END CONVENTION NOTES")
end190 = [l for l in Path("/work/notes.md").read_text().splitlines()
          if "END" in l and any(w in l.lower()
                               for w in ["convention", "numeric", "units", "exact", "verif"])]
for l190 in end190[:7]:
    print(l190[:250])

with open("/work/notes.md", "a") as f190:
    f190.write("\n## Run 190: recover training bundle reconstruction conventions\n"
               "Installed b_in matches baseline77 exactly. Loaded full records and "
               "paired train states for repaired-state detector anchoring. "
               "No weights changed. Python runs: 190; forwards: 22.\n")
```