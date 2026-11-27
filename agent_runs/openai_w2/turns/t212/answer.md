Independent record analyses support a negative insulin effect and a positive norepinephrine effect. The estimated persistence is less certain, so accumulated-effect targets remain estimates, not truth. The installed weights exactly match the validated candidate. I’ll now check its per-stay rollout readouts and identify tuning stays not yet used for factual validation. Remaining target gaps would support refinement; a factual loss near the limit would argue for restraint.

```python
import inspect

dose183 = pd.read_csv("/work/fwd_21/dose_edit_edited.csv")
print("ROLLOUT COLUMNS", list(dose183.columns))
print(dose183.head(3).to_string(index=False))
if {"stay_id", "edit", "shift"}.issubset(dose183.columns):
    reached183 = dose183.copy()
    if "reached" in reached183:
        r183 = reached183["reached"]
        reached183 = reached183[
            r183 if r183.dtype == bool else
            r183.astype(str).str.lower().isin(["true", "1", "1.0"])
        ]
    means183 = reached183.groupby(["edit", "stay_id"])["shift"].mean()
    print("\nCURRENT PER-STAY READOUTS")
    print(means183.groupby(level=0).agg(["mean", "std", "count"]).to_string())

all_tuning_ids183 = sorted(
    pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique().tolist()
)
used_ids183 = set()
for k in [20, 22]:
    fact183 = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    used_ids183.update(fact183["stay_id"].unique().tolist())
remaining_ids183 = [int(s) for s in all_tuning_ids183 if s not in used_ids183]
print("\nFACTUAL VALIDATION COVERAGE", len(used_ids183),
      "tested;", len(remaining_ids183), "remaining")
print("NEXT COHORT", remaining_ids183[:100])

print("\nTARGET HELPERS")
for name in ["linear_rollout_targets", "record_impulse_targets"]:
    obj = globals().get(name)
    if callable(obj):
        print(name, inspect.signature(obj))
        print("locals", obj.__code__.co_varnames[:28])
        print("globals", obj.__code__.co_names[:25])
    elif obj is not None:
        print(name, type(obj).__name__, getattr(obj, "shape", None))
for name in ["record_targets", "targets165", "target_ranges", "iv_persistence_table"]:
    obj = globals().get(name)
    if isinstance(obj, pd.DataFrame):
        print("\n", name, obj.shape, list(obj.columns))
        print(obj.head(5).to_string(index=False))
    elif isinstance(obj, dict):
        print(name, "keys", list(obj.keys())[:12])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Runs181–183: causal evidence recovered\n"
            "Independent sensor/history-adjusted record analyses support insulin around "
            "-1.8 to -2.0 per rate unit and NE around +3.6. Clean insulin lab threshold "
            "estimate is smaller and less precise; finger observation discontinuities "
            "warn against using naive RD. Persistence 0.9696/0.8472 comes from an "
            "assumption-dependent latent model, not known simulator truth. "
            "Current installed edit exactly equals validated_B176. Auditing per-stay "
            "readouts and expanding independent factual coverage before refinement.\n")
```