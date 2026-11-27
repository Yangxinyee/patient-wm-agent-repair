The cached weights and time-flow calculation are correct, and drug-bundle posterior states now match closely. The remaining error is confined to bundles without actions. I’ll test whether their action input should be zero or the encoded empty set. A near-exact match under one convention would identify the error without changing any model weights.

```python
r197_action = torch.tensor(
    (meta188.has_insulin | meta188.has_norepinephrine).to_numpy()[rows188],
    dtype=torch.bool
)

def posterior197(ctx197, e197, u197):
    return lin104(
        F104.gelu(lin104(
            torch.cat([ctx197, e197, u197], dim=1), "z_post.0"
        )),
        "z_post.2"
    )[:, :32]

r197_empty_action = encode_variable_sets_repair(
    [[]], "encode_action", zero_empty=False
)
r197_variants = {
    "cached": u191,
    "zero_when_no_action": torch.where(
        r197_action[:, None], u191, torch.zeros_like(u191)
    ),
    "encoded_empty_when_no_action": torch.where(
        r197_action[:, None], u191,
        r197_empty_action.expand_as(u191)
    ),
}

print("NO-ACTION INPUT NORMS")
print("cached mean:", float(u191[~r197_action].norm(dim=1).mean()))
print("encoded empty:", float(r197_empty_action.norm()))
print("\nPOSTERIOR RMS BY ACTION-EMPTY CONVENTION")
r197_results = {}
for r197_label, r197_u in r197_variants.items():
    r197_z = posterior197(ctx188, e191, r197_u)
    r197_err = (r197_z - z188).square().mean(dim=1).sqrt()
    r197_results[r197_label] = (
        float(r197_err.square().mean().sqrt()),
        float(r197_err[~r197_action].mean()),
        float(r197_err[r197_action].mean()),
        float(r197_err.max())
    )
    print(r197_label, r197_results[r197_label])

print("\nSINGLE-MEASUREMENT UNIT CHECK")
r197_single = [
    j for j, full_j in enumerate(rows188)
    if not bool(r197_action[j])
    and len(obs191[int(full_j)]) == 1
    and int(meta188.iloc[int(full_j)].n_atoms) == 1
]
r197_single = r197_single[:80]
r197_best_empty = min(r197_results, key=lambda k: r197_results[k][0])
r197_u_best = r197_variants[r197_best_empty]
r197_units_n = Wrepair["embed.categorical.unit.weight"].shape[0]
r197_unit_errors = {}
for r197_unit in range(r197_units_n):
    r197_atoms = []
    r197_positions = []
    for j in r197_single:
        full_j = int(rows188[j])
        matching = records188[
            (records188.stay_id == int(meta188.iloc[full_j].stay_id))
            & np.isclose(records188.hour, float(meta188.iloc[full_j].hour))
        ]
        if len(matching) != 1 or matching.iloc[0].event not in measurement_events:
            continue
        r197_record = matching.iloc[0].to_dict()
        r197_event = r197_record["event"]
        r197_atom = embed_record_atom_v2(r197_record)
        r197_atom = (
            r197_atom
            - Wrepair["embed.categorical.unit.weight"][unit_ids_repair[r197_event]]
            + Wrepair["embed.categorical.unit.weight"][r197_unit]
        )
        r197_atoms.append([r197_atom])
        r197_positions.append(j)
    if r197_positions:
        r197_pos = np.asarray(r197_positions)
        r197_e = encode_variable_sets_repair(r197_atoms, "encode")
        r197_z = posterior197(
            ctx188[r197_pos], r197_e, r197_u_best[r197_pos]
        )
        r197_unit_errors[r197_unit] = float(
            (r197_z - z188[r197_pos]).square().mean().sqrt()
        )
print("best empty convention:", r197_best_empty)
print("pooled single-reading unit RMS:", r197_unit_errors)
print("edit unchanged:", np.array_equal(
    load("core.b_in.weight"), validated_B176
))
```
