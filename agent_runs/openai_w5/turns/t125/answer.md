This is the same run-110 output, so it does not establish that the new candidate was built. I’ll check the kernel before proceeding. If the candidate exists, I’ll inspect its validation metrics; otherwise, I’ll rebuild it rather than assume any weights changed.

```python
print("=== CANDIDATE EXECUTION CHECK ===")
for name112 in [
    "action_candidate111", "bestaction111", "fits111",
    "reconerr111", "hn111", "hnv111", "qt111", "qv111"
]:
    obj112 = globals().get(name112)
    if obj112 is None:
        print(name112, "NOT PRESENT")
    elif name112 == "bestaction111":
        print(name112, "metrics:", obj112[3])
    elif name112 == "fits111":
        print(name112, [x[3] for x in obj112])
    elif isinstance(obj112, dict):
        print(name112, "tensor names:", list(obj112))
    else:
        print(name112, getattr(obj112, "shape", type(obj112).__name__))

print("\n=== TUNING STAY IDS ===")
print("panel index names:", panel_tune.index.names)
if "stay_id" in panel_tune.index.names:
    tuning_ids112 = sorted(set(panel_tune.index.get_level_values("stay_id")))
elif "stay_id" in panel_tune.columns:
    tuning_ids112 = sorted(panel_tune.stay_id.unique().tolist())
else:
    tuning_ids112 = []
print("count", len(tuning_ids112), "first 100:", tuning_ids112[:100])
print("\nCurrent edits:", edited())
```
