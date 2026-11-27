The kernel still contains the reconstructed posterior features and synthetic-dose probes, so we can reuse them. I’ll locate their helpers and inspect the remaining reconstruction errors. If the corrected unit mapping also works in mixed bundles, we can proceed to a dose-residual edit rather than another broad amplification.

```python
import inspect, dis

_snap99 = list(globals().items())
print("=== DICTIONARY INVENTORY ===")
print("; ".join(
    f"{n}[{len(o)}]"
    for n, o in sorted(_snap99)
    if not n.startswith("_") and isinstance(o, dict)
))

print("\n=== RELEVANT DICTIONARIES ===")
for _n99, _o99 in sorted(_snap99):
    if (isinstance(_o99, dict) and not _n99.startswith("_")
        and any(s in _n99.lower() for s in
                ["unit", "repair", "post", "state", "encoder", "numeric", "cfg", "latent"])):
        _items99 = list(_o99.items())[:8]
        print(_n99, [(k, getattr(v, "shape", str(v)[:120])) for k, v in _items99])

print("\n=== HELPER IMPLEMENTATIONS ===")
for _n99 in ["embed_record_atom_v2", "encode_variable_sets_repair",
             "previous_context_repair", "posterior_error86"]:
    _f99 = globals()[_n99]
    print("\n", _n99)
    try:
        print(inspect.getsource(_f99)[:2200])
    except Exception:
        print("Arguments/locals:", _f99.__code__.co_varnames)
        print("Global references:", _f99.__code__.co_names)
        print("Constants:", str(_f99.__code__.co_consts)[:900])

print("\n=== REMAINING POSTERIOR ERRORS ===")
print(diag95.groupby(["has_action", "events"])["error"]
      .agg(["size", "mean", "max"]).sort_values("mean", ascending=False).head(16))
print("\n=== RELEVANT WEIGHT SHAPES ===")
for _n99 in NAMES:
    if _n99.startswith(("embed.", "encode.", "encode_action.", "z_post.")):
        print(_n99, SHAPES[_n99])
print("\nCurrent edits:", edited())
```