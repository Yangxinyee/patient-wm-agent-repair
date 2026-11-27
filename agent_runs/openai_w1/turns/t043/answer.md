Most remaining error comes from END-only bundles, not active infusions. I’ll now test whether posterior features can recover centered doses on held-out stays and remain quiet at measurement bundles. Strong held-out decoding with little non-action leakage would support a small correction; otherwise I’ll reject this route.

```python
# Build posterior features for every reconstructable bundle, including readings.
r36_world = {}
for r36_row in r33_records.itertuples(index=False):
    r36_key = (int(r36_row.stay_id), round(float(r36_row.hour), 4))
    r36_world.setdefault(r36_key, []).append(r36_row)

r36_indices = []
for r36_i in range(1, len(r33_ids)):
    r36_key = (int(r33_ids[r36_i]), round(float(r33_hours[r36_i]), 4))
    r36_n = len(r36_world.get(r36_key, [])) + len(r33_ends.get(r36_key, []))
    if (r33_ids[r36_i] == r33_ids[r36_i-1]
        and 0 <= r33_hours[r36_i] <= 24
        and r36_n == int(r33_states["n_atoms"][r36_i])):
        r36_indices.append(r36_i)
r36_indices = np.asarray(r36_indices)

def r36_atom(code, value=np.nan, rate=np.nan, dose=np.nan, end=False):
    numeric = np.zeros(6, np.float32)
    bins = {"value": 0, "dose": 0, "rate": 0}
    if not end:
        for attr, val, col in [("value", value, 0), ("dose", dose, 2),
                               ("rate", rate, 4)]:
            if np.isfinite(val) and (code, attr) in r33_numeric:
                meta = r33_numeric[(code, attr)]
                numeric[col] = np.clip((val-meta["median"])/meta["iqr"], -10, 10)
                numeric[col+1] = 1
                bins[attr] = int(1 + np.searchsorted(
                    meta["fine_bin_edges"], val, side="left"))
    ekind = (2 if code in drugs else 1 if code in sensors
             else 3 if code == "hospital_admission" else 4)
    result = (r33_weights["embed.code.weight"][r33_codes[code]]
              + r33_weights["embed.kind.weight"][2 if end else 0]
              + r33_weights["embed.ekind.weight"][ekind]
              + r33_linear(torch.tensor(numeric), "embed.numeric"))
    for attr in ["value", "dose", "rate"]:
        result += r33_weights[f"embed.bins.{attr}.weight"][bins[attr]]
    if not end and code in sensors:
        result += r33_weights["embed.categorical.unit.weight"][
            2 if code.startswith("glucose") else 3]
    if not end and code in drugs:
        result += r33_weights["embed.categorical.rate_unit.weight"][
            2 if code == "insulin" else 3]
    for table in ["embed.kq.weight", "embed.text.weight"]:
        if table in r33_weights:
            result += r33_weights[table][0]
    return result

r36_vectors, r36_owners, r36_action_flags = [], [], []
r36_rates = np.zeros((len(r36_indices), 2), np.float32)
r36_has_start = np.zeros((len(r36_indices), 2), bool)
for r36_j, r36_i in enumerate(r36_indices):
    r36_key = (int(r33_ids[r36_i]), round(float(r33_hours[r36_i]), 4))
    for r36_row in r36_world.get(r36_key, []):
        r36_vectors.append(r36_atom(
            r36_row.event, r36_row.value, r36_row.rate, r36_row.dose))
        r36_owners.append(r36_j)
        r36_action_flags.append(r36_row.event in drugs)
        if r36_row.event in drugs:
            r36_d = drugs.index(r36_row.event)
            r36_rates[r36_j, r36_d] += r36_row.rate
            r36_has_start[r36_j, r36_d] = True
    for r36_code, r36_rate, r36_dose in r33_ends.get(r36_key, []):
        r36_vectors.append(r36_atom(r36_code, end=True))
        r36_owners.append(r36_j)
        r36_action_flags.append(False)
r36_vectors = torch.stack(r36_vectors)
r36_owners = torch.tensor(r36_owners)
r36_action_flags = torch.tensor(r36_action_flags)

def r36_set_encode(prefix, mask, empty_zero):
    counts = torch.zeros(len(r36_indices))
    counts.index_add_(0, r36_owners[mask], torch.ones(int(mask.sum())))
    sums = torch.zeros(len(r36_indices), 192)
    sums.index_add_(0, r36_owners[mask],
                   r33_mlp(r36_vectors[mask], prefix+".mlp1", "gelu"))
    result = r33_ln(r33_mlp(
        sums/counts.clamp_min(1).sqrt()[:, None]
        + r33_linear(counts.log1p()[:, None], prefix+".size"),
        prefix+".mlp2", "gelu"), prefix+".norm")
    if empty_zero:
        result[counts == 0] = 0
    return result

r36_e = r36_set_encode("encode", ~r36_action_flags, False)
r36_u_options = {
    "zero": r36_set_encode("encode_action", r36_action_flags, True),
    "encoded_empty": r36_set_encode("encode_action", r36_action_flags, False)}
r36_previous = torch.tensor(r33_states["x"][r36_indices-1])
r36_dt = torch.tensor(
    3600*(r33_hours[r36_indices]-r33_hours[r36_indices-1]), dtype=torch.float32)
r36_flow = r36_previous[:, 192:960].reshape(-1, 4, 192)*torch.exp(
    -r36_dt[:, None, None]*torch.exp(r33_weights["core.flow.a"])[None, None, :])
r36_ctx = (r36_previous[:, :192] + r33_ln(r36_flow.mean(1), "core.out_norm")
           + r33_mlp((r36_dt.clamp_min(1).log()/10)[:, None], "gap", "gelu"))
r36_latent_target = torch.tensor(r33_states["x"][r36_indices, -32:])
r36_recon = []
for r36_mode, r36_u in r36_u_options.items():
    r36_input_test = torch.cat([r36_ctx, r36_e, r36_u], 1)
    r36_hidden_test = F.gelu(r33_linear(r36_input_test, "z_post.0"))
    r36_prediction_test = r33_linear(r36_hidden_test, "z_post.2")[:, :32]
    r36_errors = (r36_prediction_test-r36_latent_target).square().mean(1)
    r36_recon.append((
        float(r36_errors.mean().sqrt()), r36_mode,
        float(r36_errors[r36_has_start.any(1)].mean().sqrt()),
        float(r36_errors[~r36_has_start.any(1)].mean().sqrt())))
print("Posterior reconstruction (overall, empty mode, active, non-action):")
for r36_row in r36_recon:
    print(r36_row)
r36_empty_mode = min(r36_recon)[1]
r36_u = r36_u_options[r36_empty_mode]
r36_input = torch.cat([r36_ctx, r36_e, r36_u], 1)
r36_hidden = F.gelu(r33_linear(r36_input, "z_post.0"))
print("Bundles:", len(r36_indices), "empty mode:", r36_empty_mode)

# Align previously cross-fitted record-based dose innovations to bundles.
r36_sid_to_panel = {int(s): j for j, s in enumerate(stay_ids)}
r36_sid = r33_ids[r36_indices]
r36_hour = np.floor(r33_hours[r36_indices]+1e-5).astype(int)
r36_targets = {}
for r36_drug, r36_sensor in [("insulin", "glucose_finger"),
                            ("norepinephrine", "map_arterial")]:
    r36_source = causal_residuals[(r36_sensor, r36_drug)]
    r36_mask = np.asarray(r36_source["mask"], bool)
    r36_rd = np.asarray(r36_source["rd"])
    print(r36_drug, "residual source:", r36_mask.shape, r36_rd.shape)
    if r36_mask.size != len(stay_ids)*24:
        raise ValueError("Unexpected residual alignment; do not infer indices.")
    r36_matrix = np.full(r36_mask.size, np.nan, np.float32)
    if r36_rd.size == int(r36_mask.sum()):
        r36_matrix[r36_mask.ravel()] = r36_rd
    elif r36_rd.size == r36_mask.size:
        r36_matrix[r36_mask.ravel()] = r36_rd.ravel()[r36_mask.ravel()]
    else:
        raise ValueError("Unexpected residual length.")
    r36_matrix = r36_matrix.reshape(len(stay_ids), 24)
    r36_d = drugs.index(r36_drug)
    r36_target = np.zeros(len(r36_indices), np.float32)
    for r36_j in np.flatnonzero(r36_has_start[:, r36_d]):
        r36_h = r36_hour[r36_j]
        r36_target[r36_j] = (
            r36_matrix[r36_sid_to_panel[int(r36_sid[r36_j])], r36_h]
            if 0 <= r36_h < 24 else np.nan)
    r36_targets[r36_drug] = r36_target

r36_holdout_stays = set(np.random.default_rng(36).choice(
    np.unique(r36_sid), len(np.unique(r36_sid))//2, replace=False))
r36_holdout = np.array([s in r36_holdout_stays for s in r36_sid])
r36_feature_sets = {
    "posterior_hidden": r36_hidden.numpy(),
    "posterior_input": r36_input.numpy()}
r36_decoder_results, r36_decoders = [], {}
with threadpool_limits(limits=8):
    for r36_fname, r36_features in r36_feature_sets.items():
        for r36_drug in drugs:
            r36_y = r36_targets[r36_drug]
            r36_ok = np.isfinite(r36_y)
            r36_fit = r36_ok & ~r36_holdout
            r36_active_test = (r36_ok & r36_holdout
                              & r36_has_start[:, drugs.index(r36_drug)])
            r36_inactive_test = (r36_holdout
                                & ~r36_has_start[:, drugs.index(r36_drug)])
            r36_mean = r36_features[r36_fit].mean(0)
            r36_sd = r36_features[r36_fit].std(0).clip(1e-4)
            r36_scaled = (r36_features-r36_mean)/r36_sd
            for r36_alpha in [.1, 1., 10., 100.]:
                r36_model = Ridge(alpha=r36_alpha).fit(
                    r36_scaled[r36_fit], r36_y[r36_fit])
                r36_pred = r36_model.predict(r36_scaled)
                r36_decoder_results.append({
                    "features": r36_fname, "drug": r36_drug, "alpha": r36_alpha,
                    "active_n": int(r36_active_test.sum()),
                    "active_r2": r2_score(r36_y[r36_active_test],
                                          r36_pred[r36_active_test]),
                    "inactive_rms": float(np.sqrt(np.mean(
                        r36_pred[r36_inactive_test]**2))),
                    "active_sd": float(np.std(r36_y[r36_active_test]))})
                r36_decoders[(r36_fname, r36_drug, r36_alpha)] = (
                    r36_model, r36_mean, r36_sd)
r36_decoder_table = pd.DataFrame(r36_decoder_results)
print(r36_decoder_table.to_string(index=False))

with open("/work/notes.md", "a") as r36_file:
    r36_file.write(
        "\n## Run 36\n"
        "Built posterior input/hidden features across action and measurement "
        "bundles, and tested stay-held-out decoding of record-based dose "
        "innovations, with zero targets at non-action bundles.\n"
        f"Reconstruction: {r36_recon}\n"
        + r36_decoder_table.to_string(index=False)
        + "\nCounts: 36 Python runs, 6 forward passes. No edits.\n")
```
