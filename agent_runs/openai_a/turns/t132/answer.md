This repeats the earlier inventory, so it provides no new validation result. I’ll run the focused two-reading check setup now. Matching its posterior means would support proceeding to synthetic dose contrasts; a persistent mismatch would keep that route blocked.

```python
print("=== LATENT / GAP TENSORS ===")
for n in NAMES:
    if any(s in n for s in ("z_post", "z_prior", "z_in", "z_head", "gap.")):
        print(n, SHAPES[n])

print("\n=== MEASUREMENT HELPER DETAILS ===")
for n, v in sorted(list(globals().items())):
    if n.startswith(("r103_", "r109_", "r110_", "r111_", "r96_", "_r96_")):
        if callable(v) and hasattr(v, "__code__"):
            print(n, inspect.signature(v), v.__code__.co_names,
                  repr(v.__code__.co_consts)[:550])
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            print(n, tuple(v.shape))

r117_groups = {
    (int(s), round(float(h), 4)): g
    for (s, h), g in rm_records.groupby(["stay_id", "hour"], sort=False)
}
r117_ids, r117_groups_used = [], []
for i in range(1, len(rm_states["hour"])):
    s, h = int(rm_states["stay_id"][i]), float(rm_states["hour"][i])
    if s != int(rm_states["stay_id"][i-1]) or h < 1 or abs(h-round(h)) > 1e-4:
        continue
    g = r117_groups.get((s, round(h, 4)))
    if g is not None and len(g) == 2 and set(g.event) == {"glucose_finger", "map_arterial"}:
        if int(rm_states["n_atoms"][i]) == 2:
            r117_ids.append(i)
            r117_groups_used.append(g)
    if len(r117_ids) == 48:
        break
r117_ids = np.asarray(r117_ids, dtype=int)
print("\n=== TWO-READING VALIDATION SET ===")
print("bundles:", len(r117_ids))
print("state indices:", r117_ids[:8])
print("first group:", r117_groups_used[0].to_dict("records") if len(r117_ids) else None)
```
