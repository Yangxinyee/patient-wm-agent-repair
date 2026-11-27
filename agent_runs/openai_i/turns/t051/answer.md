The remaining discrepancy may be caused by adding categorical embeddings for fields absent from an END atom. I’ll compare omitted fields with index-zero fields, and compare against the previously exact start reconstruction. This will separate an embedding convention error from a timing or state error.

```python
print("=== DEFAULT CATEGORY NORMS ===")
for field44 in ["unit", "rate_unit", "route", "action", "status", "discharged_to"]:
    tab44 = repair_weights[f"embed.categorical.{field44}.weight"]
    print(field44, [round(float(v.norm()), 6) for v in tab44])

def latent_start_atom(row):
    insulin44 = row["event"] == "insulin"
    return (repair_atom_base(row)
            + repair_weights["embed.categorical.unit.weight"][3 if insulin44 else 4]
            + repair_weights["embed.categorical.rate_unit.weight"][2 if insulin44 else 3])

u44 = torch.stack([latent_action_encode([latent_start_atom(r)])
                   for _, r in single_start41])
for label44, action44 in [("all default categories", u41), ("present fields only", u44)]:
    zz44 = latent_post_mean(latent_context(idx41),
                            latent_empty_e.expand(len(idx41), -1), action44)
    print("Start", label44, "RMS:",
          float(((zz44-torch.from_numpy(latent_state[idx41, 960:]))**2).mean().sqrt()))

end_direct44 = []
for numbers44, categories44, kind44, ekind44 in itertools.product(
        [False, True], ["none", "default", "canonical"], [0, 1, 2], [0, 2]):
    atoms44 = []
    for source44 in end_sources42:
        av44 = repair_atom_base(source44, kind=kind44, include_numbers=numbers44)
        av44 = av44 - repair_weights["embed.ekind.weight"][2] + repair_weights["embed.ekind.weight"][ekind44]
        if categories44 == "default":
            av44 = av44 + latent_categories(0, 0)
        elif categories44 == "canonical":
            ins44 = source44["event"] == "insulin"
            av44 = av44 + latent_categories(3 if ins44 else 4, 2 if ins44 else 3)
        atoms44.append(av44)
    e44 = latent_encode_batch(torch.stack(atoms44), "encode")
    z44 = latent_post_mean(latent_context(end_idx42), e44, torch.zeros_like(e44))
    err44 = float(((z44-end_target42)**2).mean().sqrt())
    end_direct44.append((err44, numbers44, categories44, kind44, ekind44))
print("END omitted-field tests:", sorted(end_direct44)[:8])

print("\n=== COMPARISON WITH RETAINED EXACT INPUTS ===")
case_idx44 = np.array([i for i, r in cases])
print("Stored/file state agreement:",
      bool(np.array_equal(state_ids, latent_sid)),
      float(np.max(np.abs(state_x-latent_state))))
if len(cases) == actions.shape[1] and np.all(
        latent_sid[case_idx44] == np.array([int(r["stay_id"]) for i, r in cases])):
    case_u44 = torch.stack([latent_action_encode([latent_start_atom(r)]) for i, r in cases])
    action_error44 = ((actions-case_u44[None])**2).mean((1, 2)).sqrt()
    print("Closest retained action encoding:",
          int(action_error44.argmin()), float(action_error44.min()))
    case_ctx44 = latent_context(case_idx44)
    for label44, stored44 in contexts.items():
        print("Context", label44, "RMS:", float(((stored44-case_ctx44)**2).mean().sqrt()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 44; forward 6 — absent-field test\n"
            f"- END omitted-field tests: {sorted(end_direct44)[:3]}.\n"
            "- Comparing new start reconstruction with retained exact inputs; no weights changed.\n")
```