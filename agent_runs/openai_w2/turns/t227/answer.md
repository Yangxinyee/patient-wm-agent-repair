The state archive and metadata appear aligned, and the helpers reference the expected unit dictionaries. That does not yet identify the error. I’ll inspect the cached inputs and compare reconstruction errors by bundle type. Errors concentrated in END bundles would implicate END handling; errors in ordinary measurement bundles would implicate embedding or context construction.

```python
print("CURRENT UNIT MAPPINGS")
print(unit_ids_repair)
print(rate_unit_ids_repair)
print("\nVOCABULARY KEYS")
print(list(VOCAB.keys()))
for r195_key in VOCAB:
    if any(s in r195_key.lower() for s in ("categor", "unit", "kind", "codes")):
        print(r195_key, str(VOCAB[r195_key])[:1400])

print("\nRECONSTRUCTION VARIABLES")
for r195_name, r195_obj in sorted(list(globals().items())):
    if (
        r195_name.endswith(("188", "189", "190", "191"))
        or r195_name in ("e_full", "u_full", "e_full134", "u_full134", "meta85", "meta92")
    ):
        if hasattr(r195_obj, "shape"):
            print(r195_name, type(r195_obj).__name__, tuple(r195_obj.shape))
        elif isinstance(r195_obj, (list, tuple, dict)):
            print(r195_name, type(r195_obj).__name__, "length", len(r195_obj))
        elif not callable(r195_obj):
            print(r195_name, str(r195_obj)[:120])

print("\nCORE HELPER DEPENDENCIES")
for r195_name in ("lin104", "norm104", "attention104", "batched_set_repair"):
    r195_f = globals().get(r195_name)
    if r195_f is not None:
        print(r195_name, r195_f.__code__.co_names,
              str(r195_f.__code__.co_consts)[:700])

print("\nMETADATA ALIGNMENT")
for r195_key in ("stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms"):
    print(r195_key, np.array_equal(meta188[r195_key].to_numpy(), S187[r195_key]))

print("\nLATEST RECONSTRUCTION ERROR BY BUNDLE")
if "err190" in globals():
    r195_report = meta188.iloc[np.flatnonzero(
        np.r_[False, S187["stay_id"][1:] == S187["stay_id"][:-1]]
    )].copy()
    if len(r195_report) == len(err190):
        r195_report["error"] = err190
        r195_report["action"] = (
            r195_report.has_insulin | r195_report.has_norepinephrine
        )
        print(r195_report.groupby("action").error.agg(["size", "mean", "max"]))
        print(r195_report.nlargest(6, "error").to_string(index=False))

print("\nOLD SAVED-ENCODING OBJECTS")
print([
    n for n in sorted(globals())
    if n.startswith(("e_", "u_", "h_", "E", "U", "ctx", "keep", "idx", "noninitial"))
    and hasattr(globals()[n], "shape")
])
```
