The graft changes only two hidden units, so further trials can remain small and reversible. For norepinephrine, I’ll examine dose initiation separately from ongoing infusions. A sharp initiation boundary with smooth prior histories could support a causal analysis; a smooth association would leave the earlier estimate unverified.

```python
import pandas as pd

p_ne = panel.copy()
p_ne["ne_prev"] = p_ne["norepinephrine"].groupby(level="stay_id").shift(1)
p_ne["map_next"] = p_ne["map_arterial"].groupby(level="stay_id").shift(-1)
p_ne["map_prev"] = p_ne["map_arterial"].groupby(level="stay_id").shift(1)
p_ne["active"] = (p_ne["norepinephrine"] > 0).astype(float)
valid_hours_ne = p_ne.index.get_level_values("t").to_numpy()
p_ne = p_ne[(valid_hours_ne >= 0) & (valid_hours_ne <= 23)]

print("=== NE DOSE DISTRIBUTION ===")
positive_ne = p_ne.loc[p_ne["norepinephrine"] > 0, "norepinephrine"]
print("Positive fraction:", float((p_ne["norepinephrine"] > 0).mean()))
print("Positive dose quantiles:", positive_ne.quantile([0, .1, .5, .9, 1]).to_dict())
print("Most common positive rates:", positive_ne.value_counts().head(12).to_dict())

print("\n=== NE INITIATION VS RECORDED PRESSURE ===")
for pressure in ("map_arterial", "map_cuff"):
    subset = p_ne[
        (p_ne["ne_prev"] == 0)
        & p_ne[pressure].between(48, 82)
    ]
    table = subset.groupby(pressure).agg(
        n=("norepinephrine", "size"),
        dose=("norepinephrine", "mean"),
        starts=("active", "mean"),
        prior_map=("map_prev", "mean"),
        next_map=("map_next", "mean"),
    )
    print(pressure)
    print(table.round(4).to_string())

print("\n=== CACHE CONTENTS ===")
for label in ("sc_fact:first", "sc_fact:tuning", "sc_syn:1.0", "sc_syn:0.5", "sc_syn:2.0"):
    obj = cache_registry_recovery.get(label)
    print(label, type(obj).__name__)
    if isinstance(obj, dict):
        print({
            k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
            for k, v in obj.items()
        })
print("NE DML cache:")
for key, obj in ne_dml_cache.items():
    print(key, type(obj).__name__,
          list(obj)[:12] if isinstance(obj, dict) else
          (getattr(obj, "shape", None)))

print("\n=== GRAFT OUTPUT PAIR ===")
w2_recovery = as_np_restore(validated_gated8_checkpoint["attr_heads.value_bin.2.weight"])
print("Pair cancellation error:",
      float(np.max(np.abs(w2_recovery[:, activity_graft_rows].sum(axis=1)))))
print("Graft implementation globals:", make_contrast_graft.__code__.co_names)
print("Cache loss implementation globals:", cache_losses.__code__.co_names)
print("Graft probability implementation globals:", graft_probs.__code__.co_names)
```