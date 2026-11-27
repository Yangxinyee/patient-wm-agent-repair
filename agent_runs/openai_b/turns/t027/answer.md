The current edit removes dose-magnitude inputs while preserving drug identity and presence. It produced exactly zero for all four rollout shifts on the first 100 tuning stays, with essentially unchanged factual NLL. I’ll now aggregate the factual checks across all 400 stays and prepare a distinct rollout cohort. Poorer fit elsewhere would refute keeping this edit.

```python
import json
import pandas as pd
import numpy as np
from pathlib import Path

print("=== REMAINING EARLY NOTES ===")
_notes = Path("/work/notes.md").read_text()
_a = _notes.find("## Run 11")
_b = _notes.find("## Run 13")
print(_notes[_a:_b][:2500] if _a >= 0 else "No Run 11 heading found.")

factual_frames = {}
for _which in ("original", "edited"):
    _frames = []
    for _k in (4, 6, 7, 8):
        _f = pd.read_csv(f"/work/fwd_{_k}/factual_{_which}.csv")
        _f["forward_file"] = _k
        _frames.append(_f)
    factual_frames[_which] = pd.concat(_frames, ignore_index=True)

print("=== FACTUAL SCHEMA ===")
print(factual_frames["original"].columns.tolist())
print(factual_frames["original"].head(2).to_string(index=False))

print("=== FORWARD SUMMARIES ===")
for _k in (4, 6, 7, 8):
    _s = json.loads(Path(f"/work/fwd_{_k}/summary.json").read_text())
    print(_k, json.dumps(_s, separators=(",", ":")))

print("=== RETAINED IDENTIFICATION CHECKS ===")
for _name in ("rd_checks", "late_checks", "magnitude_checks"):
    if _name in globals():
        print(_name, str(globals()[_name])[:1400])

validation_ids = sorted(map(int, tuning["stay_id"].unique()))
rollout_check_ids = validation_ids[100:200]
assert len(rollout_check_ids) == 100
assert len(set(rollout_check_ids)) == 100
print("=== NEXT VALID REQUEST ===")
print(json.dumps({"type": "dose_edit", "model": "both", "split": "tuning",
                  "stays": rollout_check_ids}))

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Restart / runs 16–18\n"
             "Recovered architecture and candidate A. Three input tensors edited: "
             "nonmissing dose/rate bin rows made constant; numeric magnitude columns "
             "zeroed with their mean contributions transferred to presence columns. "
             "Candidate A had exact zero dose-edit shifts on the first100 and "
             "first100 factual NLL delta approximately -0.00002. "
             "Repeated stay IDs caused a rejected forward request, not empirical evidence. "
             "Now aggregating four disjoint factual tuning batches and preparing "
             "unique IDs for a second rollout check.\n")
```
