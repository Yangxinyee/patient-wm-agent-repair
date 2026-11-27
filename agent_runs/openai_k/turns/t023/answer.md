The first-dose reconstruction passed, but bundles containing readings or infusion endings still fail. The innovation-recovery results therefore are not reliable enough to use. I’ll test observation conventions and where END atoms enter the encoders. Agreement across full bundles would validate the features; continued mismatch would block an edit based on them.

```python
rng_audit = np.random.default_rng(718)

# Find ordinary measurement bundles, whose posterior can identify input conventions.
ordinary_world_audit = np.array([
    i for i in valid_previous_audit
    if bundle_records_audit.get(state_keys_audit[i])
    and all(kind == 0 and row.event in measure_codes
            for row, kind in bundle_records_audit[state_keys_audit[i]])
])
world_sample_audit = rng_audit.choice(
    ordinary_world_audit, min(64, len(ordinary_world_audit)), replace=False)
empty_action_audit = norm0(
    mlp0(lin0(torch.zeros(1, 1), "encode_action.size"),
         "encode_action.mlp2", "gelu"), "encode_action.norm")[0]

world_candidates_audit = []
for glu_unit, map_unit, kq, bin_side in itertools.product(
        range(6), range(6), (-1, 0, 1, 2, 3), ("left", "right")):
    sets = []
    for i in world_sample_audit:
        atoms = []
        for row, kind in bundle_records_audit[state_keys_audit[i]]:
            atom = embed_record_audit(row, kind)
            new_unit = glu_unit if row.event.startswith("glucose") else map_unit
            atom = (atom - W0["embed.categorical.unit.weight"][unit_ids_audit[row.event]]
                    + W0["embed.categorical.unit.weight"][new_unit]
                    - W0["embed.kq.weight"][0])
            if kq >= 0:
                atom = atom + W0["embed.kq.weight"][kq]
            if bin_side == "right":
                spec = num_spec0[(row.event, "value")]
                b_left = 1 + np.searchsorted(spec["fine_bin_edges"], row.value, side="left")
                b_right = 1 + np.searchsorted(spec["fine_bin_edges"], row.value, side="right")
                atom = (atom - W0["embed.bins.value.weight"][b_left]
                        + W0["embed.bins.value.weight"][b_right])
            atoms.append(atom)
        sets.append(atoms)
    ee = encode_sets_audit(sets, "encode")
    for action_empty in ("zero", "encoded"):
        uu = (torch.zeros(len(ee), 192) if action_empty == "zero"
              else empty_action_audit.expand(len(ee), -1))
        pp = mlp0(torch.cat((contexts_audit[world_sample_audit], ee, uu), 1),
                  "z_post", "gelu")[:, :32]
        error = float((pp - torch.tensor(sx[world_sample_audit, 960:]))
                      .square().mean().sqrt())
        world_candidates_audit.append(
            (error, glu_unit, map_unit, kq, bin_side, action_empty))
world_candidates_audit.sort()
print("Best observation conventions:", world_candidates_audit[:5])
_, glu_unit_audit, map_unit_audit, world_kq_audit, world_bin_side_audit, action_empty_mode_audit = (
    world_candidates_audit[0])
for event in measure_codes:
    unit_ids_audit[event] = glu_unit_audit if event.startswith("glucose") else map_unit_audit

def embed_consistent_audit(row, kind, end_numeric="absent", end_categories="absent",
                           end_kq=0):
    atom = embed_record_audit(row, kind, end_numeric, end_categories)
    if kind == 0 and row.event in measure_codes:
        atom = atom - W0["embed.kq.weight"][0]
        if world_kq_audit >= 0:
            atom = atom + W0["embed.kq.weight"][world_kq_audit]
        if world_bin_side_audit == "right":
            spec = num_spec0[(row.event, "value")]
            b_left = 1 + np.searchsorted(spec["fine_bin_edges"], row.value, side="left")
            b_right = 1 + np.searchsorted(spec["fine_bin_edges"], row.value, side="right")
            atom = atom - W0["embed.bins.value.weight"][b_left] + W0["embed.bins.value.weight"][b_right]
    if kind == 2:
        atom = atom - W0["embed.kq.weight"][0] + W0["embed.kq.weight"][end_kq]
    return atom

def bundle_encodings_audit(indices, end_route, end_numeric, end_categories, end_kq=0):
    world_sets, action_sets = [], []
    for i in indices:
        ww, aa = [], []
        for row, kind in bundle_records_audit[state_keys_audit[i]]:
            atom = embed_consistent_audit(row, kind, end_numeric, end_categories, end_kq)
            if row.event in drug_codes and kind == 0:
                aa.append(atom)
            elif row.event in drug_codes and kind == 2:
                if end_route in ("actions", "both"):
                    aa.append(atom)
                if end_route in ("world", "both"):
                    ww.append(atom)
            else:
                ww.append(atom)
        world_sets.append(ww)
        action_sets.append(aa)
    ee = encode_sets_audit(world_sets, "encode")
    uu = encode_sets_audit(action_sets, "encode_action",
                           zero_empty=(action_empty_mode_audit == "zero"))
    return ee, uu

ending_rows_audit = np.array([
    i for i in action_indices_audit
    if any(kind == 2 for row, kind in bundle_records_audit[state_keys_audit[i]])
])
ending_sample_audit = rng_audit.choice(
    ending_rows_audit, min(64, len(ending_rows_audit)), replace=False)
ending_route_candidates_audit = []
for route, numeric, categories, kq in itertools.product(
        ("actions", "world", "both", "ignore"),
        ("absent", "zero", "old", "rate_zero", "rate_old"),
        ("absent", "known", "unit_only", "rate_only"), range(4)):
    ee, uu = bundle_encodings_audit(ending_sample_audit, route, numeric, categories, kq)
    pp = mlp0(torch.cat((contexts_audit[ending_sample_audit], ee, uu), 1),
              "z_post", "gelu")[:, :32]
    error = float((pp - torch.tensor(sx[ending_sample_audit, 960:]))
                  .square().mean().sqrt())
    ending_route_candidates_audit.append((error, route, numeric, categories, kq))
ending_route_candidates_audit.sort()
print("Best END routing/conventions:", ending_route_candidates_audit[:6])
_, end_route_audit, best_end_numeric_audit, best_end_categories_audit, end_kq_audit = (
    ending_route_candidates_audit[0])

all_e_audit, all_u_audit = bundle_encodings_audit(
    np.arange(len(sx)), end_route_audit, best_end_numeric_audit,
    best_end_categories_audit, end_kq_audit)
post_inputs_audit = torch.cat((contexts_audit, all_e_audit, all_u_audit), 1)
post_hidden_audit = act0(lin0(post_inputs_audit, "z_post.0"), "gelu")
all_post_means_audit = lin0(post_hidden_audit, "z_post.2")[:, :32]
posterior_errors_audit = (
    all_post_means_audit - torch.tensor(sx[:, 960:])).square().mean(1).sqrt().numpy()
print("Posterior RMS, ordinary readings:",
      np.sqrt(np.mean(posterior_errors_audit[ordinary_world_audit] ** 2)))
print("Posterior RMS, action bundles:",
      np.sqrt(np.mean(posterior_errors_audit[action_indices_audit] ** 2)))
print("Posterior RMS, all noninitial:",
      np.sqrt(np.mean(posterior_errors_audit[valid_previous_audit] ** 2)))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 18: observation and END routing audit\n"
            "- Run 17's full-bundle reconstruction failed, so its innovation recovery is not used for an edit.\n"
            f"- Best observation conventions: {world_candidates_audit[:3]}\n"
            f"- Best END conventions: {ending_route_candidates_audit[:3]}\n"
            f"- Full noninitial posterior RMS: {np.sqrt(np.mean(posterior_errors_audit[valid_previous_audit] ** 2))}\n"
            "- Python runs: 18; forward passes: 4; no weights edited.\n")
```