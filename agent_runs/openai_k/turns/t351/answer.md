The current edit is the saved 400-stay candidate, with a factual value-NLL increase of 0.00338 nats. Its insulin response is near the record-based estimate, but MAP remains under-responsive. I’ll preserve it while checking the remaining rollout batches and additional factual stays; consistent responses and loss below the limit would support submission.

```python
from pathlib import Path
import numpy as np, pandas as pd, re

_current = np.asarray(load("core.b_in.weight"), dtype=np.float32)
with np.load("/work/margin_validated400.npz", allow_pickle=False) as _a:
    print("EXACT VALIDATED WEIGHT MATCH:",
          np.array_equal(_current, _a["core_b_in"]))

_notes = Path("/work/notes.md").read_text()
_sections = re.split(r"\n(?=## )", _notes)
print("=== DYNAMIC EFFECT EVIDENCE ===")
for _s in _sections:
    if re.match(r"## Run (6|132|133|134)\b", _s):
        print(_s[:1500])

print("=== RETAINED IMPULSE ESTIMATES ===")
for _k in sorted(list(globals())):
    if not _k.startswith("_") and any(s in _k.lower() for s in
                                      ["impulse", "irf", "deconv"]):
        _v = globals()[_k]
        if isinstance(_v, np.ndarray) and _v.size <= 64:
            print(_k, _v)
        elif isinstance(_v, dict) and len(_v) <= 4:
            print(_k, repr(_v)[:1000])

print("=== REMAINING TUNING BATCHES ===")
final_tuning_batches = []
for _folder in ["fwd_40", "fwd_41"]:
    _d = pd.read_csv(Path("/work") / _folder / "factual_original.csv")
    _ids = sorted(int(x) for x in _d.stay_id.unique())
    final_tuning_batches.append(_ids)
    print(_folder, "stays:", _ids)

print("=== INDEPENDENTLY SELECTED FACTUAL TRAIN BATCHES ===")
_train_ids = pd.read_csv("/data/train.csv", usecols=["stay_id"]).stay_id.unique()
_final_rng = np.random.default_rng(20261111)
_selected_train = _final_rng.choice(_train_ids, size=400, replace=False)
final_train_batches = [
    sorted(int(x) for x in _selected_train[i:i+100]) for i in range(0, 400, 100)
]
for _i in range(2):
    print("train_batch", _i, final_train_batches[_i])

print("=== CURRENT TWO-BATCH ROLLOUT MEANS ===")
for _model in ["original", "edited"]:
    _d = pd.concat([
        pd.read_csv(f"/work/{f}/dose_edit_{_model}.csv")
        for f in ["fwd_42", "fwd_43"]
    ], ignore_index=True)
    print(_model, _d.loc[_d.reached.astype(str).str.lower().eq("true")]
          .groupby("edit").shift.agg(["mean", "count"]).to_string())

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Run 294: final validation preparation\n"
             "- Checking exact equality to the validated margin archive; no weights changed.\n"
             "- Recovered remaining tuning rollout IDs and selected 400 training stays "
             "with a fixed random seed for additional factual checks.\n"
             "- Python runs: 294; forward passes: 42.\n")
```