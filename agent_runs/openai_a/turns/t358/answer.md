The files contain 397 stays, with no duplicate or unexpected rows; factual validation covers all 400 and still passes at +0.00337 nats. The weights remain unchanged. I will check why three stays lack rollout rows and pool only the tool’s reached rows. Agreement with batch summaries will confirm the pooling; eligible missing readings would require a qualification.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== POOLED READOUTS ===")
assert present_ids293.issubset(expected_ids293)
assert not dup293.any()
if all_rollouts292.reached.dtype == bool:
    reached294 = all_rollouts292.reached
else:
    reached294 = all_rollouts292.reached.astype(str).str.lower().isin(["true", "1"])
readouts294 = all_rollouts292.loc[reached294].copy()
assert readouts294.hour.between(13, 24).all()
summary294 = readouts294.groupby(["model", "edit"]).agg(
    mean_shift=("shift", "mean"),
    n_stay_hours=("shift", "size"),
    n_stays=("stay_id", "nunique"))
print(summary294.to_string())

print("=== PER-BATCH REPRODUCTION ===")
print(readouts294.groupby(["forward", "model", "edit"])["shift"].mean().to_string())

rng294 = np.random.default_rng(294)
intervals294 = {}
for edit, group in readouts294[readouts294.model == "edited"].groupby("edit"):
    c = group.groupby("stay_id")["shift"].agg(["sum", "size"])
    ix = rng294.integers(0, len(c), (5000, len(c)))
    b = c["sum"].to_numpy()[ix].sum(axis=1) / c["size"].to_numpy()[ix].sum(axis=1)
    intervals294[edit] = np.quantile(b, [.025, .975]).tolist()
print("Edited stay-bootstrap 95% intervals:", intervals294)

print("=== RECORD SCHEMA AND OMITTED STAYS ===")
print(Path("/data/README.md").read_text())
tuning_records294 = pd.read_csv("/data/tuning.csv")
print("Columns:", tuning_records294.columns.tolist())
absent_ids294 = sorted(expected_ids293 - present_ids293)
absent_records294 = tuning_records294[
    tuning_records294.stay_id.isin(absent_ids294)]
ec294 = next((c for c in ("event", "code", "event_code")
              if c in tuning_records294), None)
hc294 = next((c for c in ("hour", "icu_hour")
              if c in tuning_records294), None)
if ec294 and hc294:
    post294 = absent_records294[absent_records294[hc294] >= 13]
    print("Absent-stay post-hour12 event counts:")
    print(post294.groupby(["stay_id", ec294]).size().to_string())

print("=== CAUSAL TARGET PARAMETERS AND CUBE METADATA ===")
for name in ("ins_target", "_target_beta", "_target_rho"):
    if name in globals():
        print(name, globals()[name])
snapshot294 = list(globals().items())
for name, obj in snapshot294:
    if isinstance(obj, np.ndarray) and obj.ndim == 3 and obj.shape[1] in (25, 26, 27):
        print("Candidate cube:", name, obj.shape)
    elif name.startswith("_target") or "matched" in name.lower():
        if isinstance(obj, (int, float, np.number)):
            print(name, obj)
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj)[:12])
        else:
            print(name, type(obj).__name__, getattr(obj, "shape", None))
    elif (isinstance(obj, (list, tuple)) and len(obj) <= 30
          and all(isinstance(x, str) for x in obj)
          and any("insulin" in x for x in obj)):
        print("Candidate column list:", name, obj)
print("Target helper constants:", f293.__code__.co_consts)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Pooled final readouts; Python 294, forwards 50/50\n"
        "- Requested all400 tuning stays; rollout CSV has397, no overlapping rows.\n"
        f"- Omitted IDs: {absent_ids294}; investigating whether target readings exist.\n"
        "- Pooled only reached stay-hours, reproducing the forward summary definition.\n"
        + summary294.to_string() + "\n"
        f"- Edited stay-bootstrap intervals: {intervals294}.\n"
        "- Weights unchanged, exactly validated amplitude093_checkpoint.\n")
```
