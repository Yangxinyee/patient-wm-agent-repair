The architecture shows that readings depend on both the recurrent state and a same-sensor last-reading register. That permits a targeted correction, but the local head reconstruction still needs checking. I’ll inspect its conventions and available features; agreement across both state samples would support using it to screen edits.

```python
import inspect

print("VOCAB KEYS", list(VOCAB))
print("CONFIG", VOCAB.get("config"))
print("HEAD SPEC", head_spec, head_layout_spec)
print("DYNAMIC ESTIMATES", dyn_estimates, dynamic_alternatives)
print("TARGET CHECKS", target_checks_A)
print("TRAIN COLUMNS", list(train.columns))
print(train.head(8).to_string(index=False))
print("PANEL SAMPLE", panel.head(5).to_string())

for label, cache in [("A", cacheA), ("train", cacheTrain)]:
    print("\nCACHE", label)
    for k, v in cache.items():
        print(k, getattr(v, "shape", None),
              v[:2] if isinstance(v, np.ndarray) and v.ndim == 1 else "")
    print(cache["df"].head(3).to_string(index=False))

print("\nSELECTED TENSORS")
for name in NAMES:
    if (name.startswith(("last_", "z_", "gap", "query_", "type_", "embed.numeric",
                         "encode_action", "core.prelude", "core.b_in", "core.delta",
                         "core.c_out", "core.out_norm", "core.flow", "core.z"))
        or "transition" in name):
        print(name, SHAPES[name])

print("\nFUNCTION SIGNATURES / SOURCE")
for fn in [reading_cache, head_parts, head_mc_layout, reconstruct_head,
           head_features_layout, mechanistic_shifts, clustered_ols]:
    print("\n", fn.__name__, inspect.signature(fn))
    try:
        print(inspect.getsource(fn)[:7000])
    except (OSError, TypeError):
        print("source unavailable; constants:", fn.__code__.co_consts[:12])

print("\nSTATE SAMPLE IDS")
for j in (3, 4, 5):
    a = np.load(f"/work/fwd_{j}/states_original.npz")
    print(j, "rows", len(a["stay_id"]), "IDs", np.unique(a["stay_id"])[:10],
          "array keys", a.files)
```
