The pooling audit stopped at an assertion before producing readouts; it made no weight changes. I will inspect file coverage and duplicate rows rather than assume every requested stay appears in the output. Disjoint, correctly structured batches would indicate an overly strict audit assumption; overlapping rows would require correcting the pooling.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

print("=== ROLLOUT COVERAGE DIAGNOSTIC ===")
expected_ids293 = set(final_tuning_ids291)
for k in sources292["dose_edit"]:
    for model in ("original", "edited"):
        d = pd.read_csv(f"/work/fwd_{k}/dose_edit_{model}.csv")
        ids = set(d.stay_id)
        print(k, model, "rows", len(d), "stays", len(ids),
              "ID range", (min(ids), max(ids)),
              "unexpected IDs", len(ids - expected_ids293))
        print("Columns:", d.columns.tolist())
        dup = d.duplicated(["stay_id", "hour", "edit"], keep=False)
        print("Duplicate stay-hour-edit rows:", int(dup.sum()))
        if dup.any():
            print(d.loc[dup].head(6).to_string(index=False))

print("Combined unique stays by model:",
      all_rollouts292.groupby("model").stay_id.nunique().to_dict())
present_ids293 = set(all_rollouts292.stay_id)
print("Requested stays absent from CSV:", sorted(expected_ids293 - present_ids293))
dup293 = all_rollouts292.duplicated(
    ["model", "stay_id", "hour", "edit"], keep=False)
print("Combined duplicate rows:", int(dup293.sum()))
if dup293.any():
    print(all_rollouts292.loc[dup293].head(8).to_string(index=False))

print("=== VERIFIED FACTUAL AGGREGATION ===")
full_factual293 = pd.concat(factual_frames292, ignore_index=True)
print("Readings/stays:", len(full_factual293), full_factual293.stay_id.nunique())
print("Original/edited/delta:",
      full_factual293.original_nll.mean(),
      full_factual293.edited_nll.mean(),
      full_factual293.delta.mean())

print("=== EDIT INTEGRITY ===")
integrity293 = {
    n: np.array_equal(numpy291(load(n)), numpy291(amplitude093_checkpoint[n]))
    for n in edited()}
print(integrity293)
assert set(integrity293) == {
    "attr_heads.value_bin.0.bias", "attr_heads.value_bin.0.weight",
    "attr_heads.value_bin.2.bias", "attr_heads.value_bin.2.weight"}
assert all(integrity293.values())

print("=== SUSTAINED EFFECT IDENTIFICATION NOTE ===")
for section in sections291:
    if section.startswith("## Sustained insulin target"):
        print(section[:3600])

print("=== TARGET HELPER ===")
f293 = globals().get("matched_causal_targets163")
if callable(f293):
    print("Signature:", inspect.signature(f293))
    print("Referenced names:", f293.__code__.co_names)
    print("Local names:", f293.__code__.co_varnames)
    for name in f293.__code__.co_names:
        obj = globals().get(name)
        if obj is not None and name not in ("np", "pd"):
            print(name, type(obj).__name__, getattr(obj, "shape", None))

target_names293 = [
    n for n in list(globals())
    if ("target" in n.lower() or "causal" in n.lower())
    and not n.endswith(("291", "292", "293"))]
print("Target variables:",
      [(n, type(globals()[n]).__name__, getattr(globals()[n], "shape", None))
       for n in target_names293[:25]])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Final audit recovery; Python 293, forwards 50/50\n"
        "- Python292 stopped at a rollout-coverage/uniqueness assertion; no edits changed.\n"
        f"- Rollout CSV unique stays: {len(present_ids293)}; "
        f"combined duplicate rows: {int(dup293.sum())}.\n"
        f"- Full tuning factual readings: {len(full_factual293)}; "
        f"paired NLL increase: {full_factual293.delta.mean():.8f}.\n"
        "- All four current tensors still exactly match amplitude093_checkpoint.\n")
```
