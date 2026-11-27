The remaining large errors are concentrated in glucose_lab bundles and final admission-end bundles; most other states are close. I’ll repeat the unit-ID test using the corrected empty-action convention and check the documented numeric scaling. A near-exact lab match would confirm a unit-table mismatch rather than an encoder problem.

```python
print("=== NUMERIC CONVENTION ===")
print(VOCAB.get("numeric_note"))
print("\n=== SINGLE-MEASUREMENT UNIT SEARCH ===")
unit_search96 = []
for event96 in measurement_events:
    jj96 = np.array([
        int(r96.j) for r96 in diag95.itertuples()
        if not r96.has_action and r96.events == event96
        and len(obs_sets_repair[int(r96.j)]) == 1
    ], dtype=int)
    if len(jj96) == 0:
        continue
    atoms96 = torch.stack([obs_sets_repair[j96][0] for j96 in jj96])
    for unit96 in range(SHAPES["embed.categorical.unit.weight"][0]):
        shift96 = (
            Wrepair["embed.categorical.unit.weight"][unit96]
            - Wrepair["embed.categorical.unit.weight"][unit_ids_repair[event96]]
        )
        e96 = encode_variable_sets_repair(
            [[a96 + shift96] for a96 in atoms96], "encode")
        p96 = rmlp(torch.cat(
            [posterior_ctx_repair[jj96], e96, posterior_u_correct94[jj96]], 1),
            "z_post")[:, :32]
        er96 = (p96 - truth91[jj96]).square().mean(1).sqrt()
        unit_search96.append({
            "event": event96, "unit": unit96, "n": len(jj96),
            "mean_error": float(er96.mean()), "max_error": float(er96.max())
        })
unit_table96 = pd.DataFrame(unit_search96)
print(unit_table96.sort_values(["event", "mean_error"]).to_string(index=False))

print("\n=== CURRENT NUMERIC IMPLEMENTATION ===")
ins96 = list(dis.get_instructions(embed_record_atom))
positions96 = [i96 for i96, x96 in enumerate(ins96)
               if x96.argval in ("numeric_specs", "median", "iqr")]
if positions96:
    lo96, hi96 = max(0, min(positions96) - 3), min(len(ins96), max(positions96) + 22)
    for x96 in ins96[lo96:hi96]:
        print(x96.opname, x96.argrepr)
print("Current edits:", edited())
```
