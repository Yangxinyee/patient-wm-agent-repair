This output repeats the previous diagnostics, so the baseline rollout summary and candidate check are still missing. I’ll retrieve them now, along with the state-array interfaces. If the original rollout effects are much smaller than the record-based targets, that will support testing a latent-update repair rather than another head-only fit.

```python
from pathlib import Path

print("=== ORIGINAL ROLLOUT SUMMARY ===")
for p in sorted(Path("/work").glob("fwd_*/dose_edit*.csv")):
    t87 = pd.read_csv(p)
    selected87 = t87[
        (t87["reached"].astype(str).str.lower() == "true")
        & t87["hour"].between(13, 24)
    ]
    perstay87 = selected87.groupby(["edit", "stay_id"])["shift"].mean()
    print(p.name)
    print(perstay87.groupby("edit").agg(["mean", "std", "count"]).to_string())

print("\n=== FIT CANDIDATE ===")
c87 = candidate_edits83[1.0]
print(type(c87).__name__, getattr(c87, "shape", None))
if isinstance(c87, dict):
    print({k: (type(v).__name__, getattr(v, "shape", None)) for k, v in c87.items()})
elif isinstance(c87, (tuple, list)):
    print([(type(v).__name__, getattr(v, "shape", None)) for v in c87])
else:
    print("evaluate:", evaluate83(c87))

print("\n=== LOCAL WEIGHT LOADER ===")
print("w63:", w63.__code__.co_names, w63.__code__.co_consts)
print("bundle_inputs73 globals:", bundle_inputs73.__code__.co_names)

print("\n=== AVAILABLE STATE ARRAYS ===")
for name in ["sx63", "e_valid60", "u_valid60", "pi73",
             "rates65", "doses65", "drugs65", "owners72", "counts72",
             "events74", "dtread73", "const_train67"]:
    if name in globals():
        obj87 = globals()[name]
        print(name, type(obj87).__name__, getattr(obj87, "shape", None))
        if isinstance(obj87, dict):
            print("keys:", list(obj87)[:15])
        elif name in ["rates65", "doses65", "drugs65", "pi73"]:
            print("sample:", np.asarray(obj87)[:8])

print("\n=== COUNTERFACTUAL CONSTRUCTION ===")
print(compact_bytecode71(make_cf81, limit=3200))
print("\n=== FINAL-HEAD EVALUATOR ===")
print(compact_bytecode71(evaluate83, limit=2300))
```