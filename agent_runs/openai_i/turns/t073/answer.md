This repeats run 64’s output, so the full-bundle check has not been verified yet. I’ll run that check now, reusing its results if it already completed. Small errors across drug, reading and END bundles would confirm the features are usable; a bundle-specific mismatch would refute that.

```python
if "hidden_all65" not in globals():
    torch.set_num_threads(8)

    def mlp65(x, prefix):
        return repair_linear(F.gelu(repair_linear(x, prefix + ".0")), prefix + ".2")

    def encode_grouped65(atoms, indices, n, prefix):
        counts = torch.zeros(n)
        sums = torch.zeros(n, 192)
        if atoms:
            atoms = torch.stack(atoms)
            indices = torch.as_tensor(indices, dtype=torch.long)
            counts.index_add_(0, indices, torch.ones(len(indices)))
            for start in range(0, len(indices), 1024):
                stop = start + 1024
                sums.index_add_(0, indices[start:stop],
                                mlp65(atoms[start:stop], prefix + ".mlp1"))
        pooled = sums / counts.clamp_min(1).sqrt()[:, None]
        pooled += repair_linear(torch.log1p(counts)[:, None], prefix + ".size")
        return repair_norm(mlp65(pooled, prefix + ".mlp2"), prefix + ".norm"), counts

    bundle_lookup65 = {
        (int(s), int(round(float(t) * 60))): i
        for i, (s, t) in enumerate(zip(state_ids, state_hours))
    }
    obs_atoms65, obs_indices65, act_atoms65, act_indices65 = [], [], [], []
    bundle_rows65 = [[] for _ in state_ids]
    measurement_units65 = {
        "glucose_finger": 0, "glucose_lab": 5,
        "map_arterial": 2, "map_cuff": 2
    }
    drug_units65 = {"insulin": 3, "norepinephrine": 4}
    drug_rate_units65 = {"insulin": 2, "norepinephrine": 3}

    for row in latent_records.to_dict("records"):
        event = row["event"]
        key = (int(row["stay_id"]), int(round(float(row["hour"]) * 60)))
        i = bundle_lookup65.get(key)
        if i is not None:
            bundle_rows65[i].append((0, row))
            if event in drug_units65:
                act_atoms65.append(complete_atom57(
                    row, kind=0, ekind=2, unit=drug_units65[event],
                    rate_unit=drug_rate_units65[event], numbers=True
                ))
                act_indices65.append(i)
            else:
                obs_atoms65.append(complete_atom57(
                    row, kind=0, unit=measurement_units65.get(event, 0),
                    rate_unit=0, numbers=event in measurement_units65
                ))
                obs_indices65.append(i)
        if event in drug_units65 and pd.notna(row.get("end_hour")):
            key = (int(row["stay_id"]), int(round(float(row["end_hour"]) * 60)))
            j = bundle_lookup65.get(key)
            if j is not None:
                bundle_rows65[j].append((2, row))
                obs_atoms65.append(complete_atom57(
                    row, kind=2, ekind=2, unit=0, rate_unit=0, numbers=False
                ))
                obs_indices65.append(j)

    e_all65, e_counts65 = encode_grouped65(
        obs_atoms65, obs_indices65, len(state_ids), "encode")
    u_all65, u_counts65 = encode_grouped65(
        act_atoms65, act_indices65, len(state_ids), "encode_action")
    with np.load("/work/fwd_3/states_original.npz") as archive:
        expected_counts65 = archive["n_atoms"].copy()
    count_errors65 = np.flatnonzero(
        (e_counts65 + u_counts65).numpy() != expected_counts65
    )

    valid_indices65 = np.flatnonzero(
        np.r_[False, state_ids[1:] == state_ids[:-1]])
    contexts65 = latent_context(valid_indices65)
    target_z65 = torch.as_tensor(
        latent_state[valid_indices65, -32:], dtype=torch.float32)
    checks65, feature_versions65 = [], {}
    for empty_e65 in ["encoded", "zero"]:
        ee65 = e_all65.clone()
        if empty_e65 == "zero":
            ee65[e_counts65 == 0] = 0
        for empty_u65 in ["encoded", "zero"]:
            uu65 = u_all65.clone()
            if empty_u65 == "zero":
                uu65[u_counts65 == 0] = 0
            hidden65 = F.gelu(repair_linear(torch.cat(
                [contexts65, ee65[valid_indices65], uu65[valid_indices65]], dim=1
            ), "z_post.0"))
            pred65 = repair_linear(hidden65, "z_post.2")[:, :32]
            err65 = torch.sqrt(((pred65 - target_z65) ** 2).mean(1)).numpy()
            checks65.append({
                "empty_e": empty_e65, "empty_u": empty_u65,
                "mean_RMS": float(err65.mean()),
                "median_RMS": float(np.median(err65)),
                "p95_RMS": float(np.quantile(err65, .95))
            })
            feature_versions65[(empty_e65, empty_u65)] = (ee65, uu65, hidden65, err65)
    check_table65 = pd.DataFrame(checks65).sort_values("mean_RMS")
    best_key65 = tuple(check_table65.iloc[0][["empty_e", "empty_u"]])
    e_all65, u_all65, hidden_all65, errors_all65 = feature_versions65[best_key65]

    labels65 = []
    for i in valid_indices65:
        kinds = [kind for kind, row in bundle_rows65[i]]
        events = [row["event"] for kind, row in bundle_rows65[i] if kind == 0]
        labels65.append(
            "action+END" if u_counts65[i] > 0 and 2 in kinds else
            "action" if u_counts65[i] > 0 else
            "reading" if any(v in measurement_units65 for v in events) else
            "END" if 2 in kinds else "visit"
        )

print("Bundles:", len(state_ids), "atom-count mismatches:", len(count_errors65))
print(check_table65.to_string(index=False))
print("\nPosterior RMS by bundle type:")
print(pd.DataFrame({"type": labels65, "RMS": errors_all65})
      .groupby("type")["RMS"].agg(["count", "mean", "median", "max"]).to_string())
print("Chosen empty encoding:", best_key65, "Current edits:", edited())

marker65 = "## Full-bundle posterior reconstruction"
if marker65 not in Path("/work/notes.md").read_text():
    with open("/work/notes.md", "a") as f:
        f.write(
            "\n" + marker65 + "\n"
            + f"- Atom-count mismatches: {len(count_errors65)}.\n"
            + check_table65.to_string(index=False) + "\n"
            + "- No weights changed; checking full-bundle features before a low-rank edit.\n"
        )
```