The architecture confirms that actions enter through a separate encoder, but infusion-end atoms may also contribute to its input. Omitting those atoms could explain the reconstruction error. I’ll compare exported bundle counts with the records and inspect the embedding conventions; matching counts and posterior means would support that explanation.

```python
import pandas as pd, inspect, dis

print("=== VOCAB CONVENTIONS ===")
print("top-level keys:", list(VOCAB))
for key, val in VOCAB.items():
    if key not in ("value_bin_edges", "value_bin_centres",
                   "rate_bin_edges", "rate_bin_centres",
                   "dose_bin_edges", "dose_bin_centres"):
        text = repr(val)
        print(key, text[:11000])

print("\n=== DATAFRAMES IN KERNEL ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, pd.DataFrame):
        print(name, obj.shape, list(obj.columns))

print("\n=== ENCODING/CONTEXT FUNCTIONS ===")
for name in ("contexts37", "linear37", "mlp37", "norm37"):
    fn = globals()[name]
    try:
        print(name, inspect.getsource(fn))
    except (OSError, TypeError):
        print(name, "constants:", fn.__code__.co_consts,
              "names:", fn.__code__.co_names)
        if name == "contexts37":
            dis.dis(fn)

s40 = np.load("/work/fwd_4/states_original.npz")
raw_candidates40 = [
    (name, obj) for name, obj in list(globals().items())
    if isinstance(obj, pd.DataFrame)
    and {"stay_id", "hour", "event", "rate", "end_hour"}.issubset(obj.columns)
    and len(obj) > 100000
]
if raw_candidates40:
    raw40 = max(raw_candidates40, key=lambda p: len(p[1]))[1]
else:
    raw40 = pd.read_csv("/data/train.csv")

ids40 = np.unique(s40["stay_id"])[:2]
records40 = raw40[raw40.stay_id.isin(ids40)].copy()
print("\n=== TWO RAW STAYS ===")
print(records40.head(90).to_string(index=False))

bundle40 = pd.DataFrame({
    k: s40[k] for k in ("stay_id", "hour", "has_insulin",
                       "has_norepinephrine", "n_atoms")
})
print("\n=== EXPORTED BUNDLE COUNTS ===")
print(bundle40[bundle40.stay_id.isin(ids40)].head(75).to_string(index=False))
print("\n=== CANDIDATE OBJECT STRUCTURE ===")
for drug, candidates in action_candidates37.items():
    print(drug, type(candidates), "length", len(candidates))
    print(repr(candidates[:2])[:3500] if isinstance(candidates, list)
          else repr(candidates)[:3500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 40\nArchitecture explicitly includes END atoms for infusion records. "
            "Checking whether omitted infusion ends explain the action encoder mismatch; "
            "also inspecting vocabulary conventions and exact context computation. No edits.\n")
```