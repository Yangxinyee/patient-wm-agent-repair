The saved estimates agree on the directions—insulin lowers glucose and norepinephrine raises MAP—but differ on persistence. That uncertainty matters for multi-hour rollouts. The large target table obscured the implementation details, so I’ll print compact diagnostics; the next check is whether the reconstruction matches GPU predictions closely enough to screen weight edits.

```python
import inspect

print("TARGET MEANS", {str(k): (round(v[0], 4), v[1])
                       for k, v in target_checks_A.items()})
print("HEAD CONVENTIONS", head_spec, head_layout_spec)
print("TRAIN SAMPLE\n", train.head(4).to_string(index=False))
print("PANEL INDEX", panel.index.names, "columns", list(panel.columns))
for label, cache in [("A", cacheA), ("train", cacheTrain)]:
    print("CACHE", label,
          {k: getattr(v, "shape", type(v).__name__) for k, v in cache.items()})
    print(cache["df"].head(2).to_string(index=False))
print("KEY TENSOR SHAPES", {n: SHAPES[n] for n in NAMES
      if n.startswith(("last_", "z_", "query_", "type_", "embed.numeric",
                       "encode_action", "core.b_in", "core.delta", "core.flow"))})
for j in (3, 4, 5):
    a = np.load(f"/work/fwd_{j}/states_original.npz")
    ids = np.unique(a["stay_id"])
    print("STATES", j, "rows", len(a["stay_id"]),
          "ID range", (int(ids.min()), int(ids.max())), "keys", a.files)
print("FUNCTION SIGNATURES")
for fn in [lin0, norm0, act0, mlp0, reading_cache, head_parts, head_mc_layout,
           reconstruct_head, head_features_layout, mechanistic_shifts, clustered_ols]:
    print(fn.__name__, inspect.signature(fn))
for fn in [head_features_layout, reconstruct_head, reading_cache]:
    print("\nSOURCE", fn.__name__)
    try:
        print(inspect.getsource(fn)[:2400])
    except (OSError, TypeError):
        print("constants:", repr(fn.__code__.co_consts)[:1800])
```
