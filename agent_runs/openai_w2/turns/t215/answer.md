Across 299 stays, the factual loss increase is 0.00433 ± 0.00092 nats (cluster standard error), leaving little margin. Losses occur across sensors, not just in one extreme stay. I’ll inspect the decoder pipeline to see whether it can be fitted on more training states. A more stable decoder with lower factual disturbance could improve safety without weakening the drug response.

```python
print("FEATURE-PIPELINE FUNCTIONS")
for name in ["previous_context_repair", "_f100", "prelude_features111",
             "embed_record_atom_v2", "encode_variable_sets_repair",
             "fold_decoder147", "report_decoder147", "fit_zero_fact171"]:
    obj = globals().get(name)
    if callable(obj) and hasattr(obj, "__code__"):
        print("\n", name, inspect.signature(obj))
        print("locals:", obj.__code__.co_varnames)
        print("globals:", obj.__code__.co_names)

print("\nFEATURE-PIPELINE DATA")
for name in sorted(list(globals())):
    if re.search(r"^(S|F|E\d*|Xcheck171|X147|meta|bundle|.*repair|.*147|.*134|.*172)$", name):
        obj = globals()[name]
        if callable(obj):
            continue
        if isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:12])
        elif hasattr(obj, "files"):
            print(name, "npz", obj.files)
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape,
                  list(obj.columns)[:10] if hasattr(obj, "columns") else "")
        elif isinstance(obj, (list, tuple, str)):
            print(name, type(obj).__name__, str(obj)[:100])

print("\nAVAILABLE STATE FILES")
state_paths185 = sorted(Path("/work").glob("fwd_*/states_original.npz"))
old_train_ids185 = set()
for path in state_paths185:
    with np.load(path) as zz:
        print(str(path), zz.files)
        if "stay_id" in zz:
            ids = np.unique(zz["stay_id"])
            old_train_ids185.update(int(s) for s in ids if int(s) < 2020000000)

train_ids185 = pd.read_csv("/data/train.csv", usecols=["stay_id"])["stay_id"].unique()
available185 = np.array([int(s) for s in train_ids185 if int(s) not in old_train_ids185])
rng185 = np.random.default_rng(185)
new_train_ids185 = sorted(rng185.choice(available185, size=100, replace=False).tolist())
print("\nNEW INDEPENDENT TRAINING-STATE COHORT", new_train_ids185)
```