The END changes alone did not resolve the mismatch. Large errors also occur in single-reading bundles, suggesting a WORLD-atom embedding problem. I’ll audit measurement unit mappings and reuse the existing END helper. A unit choice that reproduces the stored posterior states would confirm that diagnosis; otherwise I’ll broaden the audit.

```python
print("END helper globals:", embed_end_repair.__code__.co_names)
print("END helper constants:", embed_end_repair.__code__.co_consts)
print("Numeric-spec keys:", list(numeric_specs)[:12])

print("\nUNIT-RELATED NOTE EXCERPTS")
paragraphs192 = Path("/work/notes.md").read_text().split("\n\n")
selected192 = [
    p for p in paragraphs192
    if any(k in p.lower() for k in
           ["unit id", "unit-id", "unit mapping", "units corrected",
            "run132", "run133", "run134", "run 132", "run 133", "run 134"])
]
print("\n\n".join(selected192)[-6500:])

print("\nUNIT DICTIONARIES")
for name, obj in list(globals().items()):
    if isinstance(obj, dict) and "unit" in name.lower() and len(obj) < 30:
        print(name, obj)

starts192, ends192 = {}, {}
for row in records188.to_dict("records"):
    key = (int(row["stay_id"]), int(round(float(row["hour"]) * 60)))
    starts192.setdefault(key, []).append(row)
    if pd.notna(row["end_hour"]):
        key_end = (int(row["stay_id"]),
                   int(round(float(row["end_hour"]) * 60)))
        ends192.setdefault(key_end, []).append(row)

unit_backup192 = unit_ids_repair.copy()
unit_results192 = {}
try:
    for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        pairs = []
        for j, i in enumerate(rows188):
            row = meta188.iloc[i]
            key = (int(row.stay_id), int(round(float(row.hour)*60)))
            world = starts192.get(key, [])
            if (len(world) == 1 and world[0]["event"] == event
                    and not ends192.get(key)):
                pairs.append((j, world[0]))
        pairs = pairs[:200]
        if not pairs:
            print(event, "has no isolated WORLD bundles")
            continue
        jj = np.array([j for j, r in pairs])
        results = []
        for unit in range(Wrepair["embed.categorical.unit.weight"].shape[0]):
            unit_ids_repair[event] = unit
            atoms = torch.stack([
                embed_record_atom_v2(r, 0, "same") for j, r in pairs])
            ee = batched_set_repair(atoms[:, None, :], "encode")
            zz = rmlp(torch.cat(
                [ctx188[jj], ee, torch.zeros_like(ee)], dim=1), "z_post")[:, :32]
            error = float(torch.sqrt(torch.mean((zz-z188[jj])**2)))
            results.append((unit, error))
        unit_results192[event] = sorted(results, key=lambda x: x[1])
        print(event, "n =", len(pairs), "unit/RMS:",
              unit_results192[event])
finally:
    unit_ids_repair.clear()
    unit_ids_repair.update(unit_backup192)

print("\nCURRENT EDIT UNCHANGED:",
      np.array_equal(np.asarray(load("core.b_in.weight")),
                     validated_B176))
```