The recovered mapping still assigns the wrong units to both glucose channels; the MAP units are already correct. I’ll correct those two entries and inspect only the reconstruction helpers. Agreement on all saved bundles—not just single measurements—is the check needed before editing the posterior.

```python
unit_ids_repair.update({
    "glucose_finger": 0,
    "glucose_lab": 5,
    "map_arterial": 2,
    "map_cuff": 2,
})
print("Corrected units:", unit_ids_repair)

print("\n=== HELPER METADATA ===")
for _n100 in ["embed_record_atom_v2", "embed_end_repair",
              "encode_variable_sets_repair", "previous_context_repair"]:
    _f100 = globals()[_n100]
    print("\n", _n100)
    try:
        print(inspect.getsource(_f100)[:1800])
    except Exception:
        print("locals:", _f100.__code__.co_varnames)
        print("globals:", _f100.__code__.co_names)
        print("constants:", str(_f100.__code__.co_consts)[:550])

print("\n=== EXISTING TORCH FEATURES ===")
for _n100, _o100 in sorted(list(globals().items())):
    if not _n100.startswith("_") and isinstance(_o100, torch.Tensor):
        print(_n100, tuple(_o100.shape))

print("\n=== BUNDLE METADATA ===")
print(meta85.head(6).to_string(index=False))
print("posterior indices:", posterior_rows_repair[:6])
print("raw group sample:", list(raw_groups94.items())[:1])
print("observation group sample:", list(world_lookup_repair.items())[:1])
print("END group sample:", list(end_lookup_repair.items())[:1])
print("\nCurrent edits:", edited())
```