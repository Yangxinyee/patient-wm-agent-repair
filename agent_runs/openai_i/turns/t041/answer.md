The scaling edit is now fully reverted. The next hypothesis is that the posterior can separate dose variation from the condition that prompted treatment. I’ll check its input encoding against the exposed original latent states before fitting anything. Close reconstruction would make a posterior edit interpretable; a mismatch would require fixing the reconstruction first.

```python
print("=== COMPACT VOCAB METADATA ===")
print("Top-level keys:", list(VOCAB.keys()))
for _key, _val in VOCAB.items():
    if _key not in (
        "config", "codes", "numeric", "value_bin_centres",
        "rate_bin_centres", "dose_bin_centres"
    ):
        print(_key, repr(_val)[:900])

print("\n=== EMBEDDING TABLE ROW NORMS ===")
for _name in NAMES:
    if _name.startswith("embed.") and _name.endswith(".weight") and len(SHAPES[_name]) == 2:
        if SHAPES[_name][0] <= 17:
            _w = np.asarray(load_original(_name))
            print(_name, np.round(np.linalg.norm(_w, axis=1), 3).tolist())

print("\n=== HELPER SIGNATURES WITHOUT DEFAULT VALUES ===")
for _name in ("linear_named", "norm_named", "mlp_named", "head_exact", "make_panel"):
    _fn = globals()[_name]
    print(_name, list(inspect.signature(_fn).parameters),
          "locals:", _fn.__code__.co_varnames[:15],
          "small constants:", [c for c in _fn.__code__.co_consts
                              if isinstance(c, (str, int, float))][:20])

print("\n=== DATA SOURCES ===")
for _name, _obj in list(globals().items()):
    if isinstance(_obj, pd.DataFrame) and {"stay_id", "hour", "event"}.issubset(_obj.columns):
        print(_name, _obj.shape, "first stay", _obj["stay_id"].iloc[0])
print("Panel index:", panel.index.names, panel.index[:3].tolist())
print("Dose residuals:", {
    k: (type(v).__name__, getattr(v, "shape", None))
    for k, v in dose_residuals.items()
})
print("Dose model folds:", {
    k: {str(kk): type(vv).__name__ for kk, vv in v.items()}
    for k, v in dose_models.items()
})

print("\n=== FIRST ORIGINAL-STAY BUNDLES ===")
_sid = int(state_ids[0])
_smask = state_ids == _sid
print(pd.DataFrame({
    "hour": state_hours[_smask],
    "insulin": state_pack["has_insulin"][_smask],
    "ne": state_pack["has_norepinephrine"][_smask],
    "atoms": state_pack["n_atoms"][_smask],
}).head(14).to_string(index=False))

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Python 34; forward 6\n"
        "- No edits. Checking exact posterior inputs against exposed original latent means.\n"
        "- A dose-innovation correction will only be considered after held-out "
        "decoding and reconstruction checks, not from dose associations alone.\n"
    )
```