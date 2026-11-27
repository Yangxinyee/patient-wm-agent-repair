The reconstruction now matches both drugs’ saved posterior means to about five millionths. Empty observation bundles were encoded rather than zeroed; strict bins and the missing categorical terms also mattered. I’ll extend the check to infusion endings and full bundles, then test whether posterior features retain dose innovations. Accurate reconstruction and held-out recovery would support a centered edit.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Reconstruct all input atoms, including automatically generated END atoms.
state_ids_audit = np.unique(train_states0["stay_id"])
record_rows_audit = df[df.stay_id.isin(state_ids_audit)]
bundle_records_audit = {}
for row in record_rows_audit.itertuples(index=False):
    key = (int(row.stay_id), round(float(row.hour), 4))
    bundle_records_audit.setdefault(key, []).append((row, 0))
    if pd.notna(row.end_hour):
        end_key = (int(row.stay_id), round(float(row.end_hour), 4))
        bundle_records_audit.setdefault(end_key, []).append((row, 2))

unit_ids_audit = {
    "glucose_finger": 2, "glucose_lab": 2,
    "map_arterial": 5, "map_cuff": 5,
    "insulin": 3, "norepinephrine": 4
}
rate_unit_ids_audit = {"insulin": 2, "norepinephrine": 3}
event_kind_ids_audit = {
    **{event: 1 for event in measure_codes},
    **{event: 2 for event in drug_codes},
    "hospital_admission": 3, "icu_admission": 4
}

def embed_record_audit(row, kind, end_numeric="absent", end_categories="absent"):
    event = row.event
    nums = np.zeros(6, np.float32)
    bin_ids = {"value": 0, "dose": 0, "rate": 0}
    for attribute, zcol, hascol in (("value", 0, 1), ("dose", 2, 3), ("rate", 4, 5)):
        value = getattr(row, attribute)
        if kind == 2:
            if end_numeric == "absent":
                value = np.nan
            elif end_numeric == "zero":
                value = 0.0 if pd.notna(value) else np.nan
            elif end_numeric == "rate_zero":
                value = 0.0 if attribute == "rate" and event in drug_codes else np.nan
            elif end_numeric == "rate_old":
                value = value if attribute == "rate" else np.nan
        if pd.notna(value) and (event, attribute) in num_spec0:
            spec = num_spec0[(event, attribute)]
            nums[zcol] = np.clip((value - spec["median"]) / spec["iqr"], -10, 10)
            nums[hascol] = 1
            bin_ids[attribute] = 1 + np.searchsorted(spec["fine_bin_edges"], value, side="left")
    atom = (W0["embed.code.weight"][code_id0[event]]
            + W0["embed.kind.weight"][kind]
            + W0["embed.ekind.weight"][event_kind_ids_audit[event]]
            + W0["embed.kq.weight"][0]
            + lin0(torch.tensor(nums), "embed.numeric"))
    for attribute in ("value", "dose", "rate"):
        atom = atom + W0["embed.bins." + attribute + ".weight"][bin_ids[attribute]]
    if kind == 0 or end_categories in ("known", "unit_only"):
        atom = atom + W0["embed.categorical.unit.weight"][unit_ids_audit.get(event, 0)]
    if kind == 0 or end_categories in ("known", "rate_only"):
        atom = atom + W0["embed.categorical.rate_unit.weight"][rate_unit_ids_audit.get(event, 0)]
    return atom

def encode_sets_audit(atom_sets, prefix, zero_empty=False):
    counts = np.array([len(atoms) for atoms in atom_sets], dtype=np.int64)
    pooled = torch.zeros(len(atom_sets), 192)
    if counts.sum():
        atoms = torch.stack([atom for atoms in atom_sets for atom in atoms])
        owners = torch.tensor(np.repeat(np.arange(len(atom_sets)), counts))
        pooled.index_add_(0, owners, mlp0(atoms, prefix + ".mlp1", "gelu"))
        pooled /= torch.tensor(np.sqrt(np.maximum(counts, 1)), dtype=torch.float32)[:, None]
    size = lin0(torch.tensor(np.log1p(counts), dtype=torch.float32)[:, None], prefix + ".size")
    encoded = norm0(mlp0(pooled + size, prefix + ".mlp2", "gelu"), prefix + ".norm")
    if zero_empty:
        encoded[counts == 0] = 0
    return encoded

state_keys_audit = [
    (int(sid), round(float(hour), 4))
    for sid, hour in zip(train_states0["stay_id"], train_states0["hour"])
]
counts_reconstructed_audit = np.array([
    len(bundle_records_audit.get(key, [])) for key in state_keys_audit
])
print("Bundle-count agreement:",
      np.mean(counts_reconstructed_audit == train_states0["n_atoms"]))

valid_previous_audit = np.flatnonzero(
    (np.arange(len(sx)) > 0)
    & (train_states0["stay_id"] == np.roll(train_states0["stay_id"], 1))
)
contexts_audit = torch.zeros(len(sx), 192)
dt_all_audit = np.zeros(len(sx), np.float32)
dt_all_audit[valid_previous_audit] = (
    train_states0["hour"][valid_previous_audit]
    - train_states0["hour"][valid_previous_audit - 1]
) * 3600
prev_audit = torch.tensor(sx[valid_previous_audit - 1])
flow_h_audit = (
    prev_audit[:, 192:960].reshape(-1, 4, 192)
    * torch.exp(-torch.tensor(dt_all_audit[valid_previous_audit])[:, None, None]
                * torch.exp(W0["core.flow.a"])[None, None, :])
)
contexts_audit[valid_previous_audit] = (
    prev_audit[:, :192] + norm0(flow_h_audit.mean(1), "core.out_norm")
    + mlp0(torch.log(torch.tensor(dt_all_audit[valid_previous_audit]).clamp_min(1))[:, None] / 10,
           "gap", "gelu")
)

action_indices_audit = np.flatnonzero(
    train_states0["has_insulin"] | train_states0["has_norepinephrine"]
)
ending_candidates_audit = []
for end_numeric, end_categories in itertools.product(
        ("absent", "zero", "old", "rate_zero", "rate_old"),
        ("absent", "known", "unit_only", "rate_only")):
    atom_sets = [
        [embed_record_audit(row, kind, end_numeric, end_categories)
         for row, kind in bundle_records_audit[state_keys_audit[i]]
         if row.event in drug_codes]
        for i in action_indices_audit
    ]
    encoded_actions = encode_sets_audit(atom_sets, "encode_action", zero_empty=True)
    e_empty = empty_world_audit["empty_gelu"].expand(len(action_indices_audit), -1)
    inputs = torch.cat((contexts_audit[action_indices_audit], e_empty, encoded_actions), 1)
    means = mlp0(inputs, "z_post", "gelu")[:, :32]
    error = float((means - torch.tensor(sx[action_indices_audit, 960:]))
                  .square().mean().sqrt())
    ending_candidates_audit.append((error, end_numeric, end_categories))
ending_candidates_audit.sort()
print("Best END conventions:", ending_candidates_audit[:5])
best_end_numeric_audit, best_end_categories_audit = ending_candidates_audit[0][1:]

world_sets_audit, action_sets_audit = [], []
for key in state_keys_audit:
    world_atoms, action_atoms = [], []
    for row, kind in bundle_records_audit[key]:
        atom = embed_record_audit(row, kind, best_end_numeric_audit, best_end_categories_audit)
        (action_atoms if row.event in drug_codes else world_atoms).append(atom)
    world_sets_audit.append(world_atoms)
    action_sets_audit.append(action_atoms)
all_e_audit = encode_sets_audit(world_sets_audit, "encode")
all_u_audit = encode_sets_audit(action_sets_audit, "encode_action", zero_empty=True)
post_inputs_audit = torch.cat((contexts_audit, all_e_audit, all_u_audit), 1)
post_hidden_audit = act0(lin0(post_inputs_audit, "z_post.0"), "gelu")
all_post_means_audit = lin0(post_hidden_audit, "z_post.2")[:, :32]
print("All noninitial posterior reconstruction RMS:",
      float((all_post_means_audit[valid_previous_audit]
             - torch.tensor(sx[valid_previous_audit, 960:])).square().mean().sqrt()))

# Held-stay-out recovery: can the posterior features express a centered dose correction?
panel_positions_audit = P.index.get_indexer([
    (int(sid), int(np.floor(hour + 1e-6)))
    for sid, hour in zip(train_states0["stay_id"], train_states0["hour"])
])
causal_positions_audit = np.full(len(sx), -1, dtype=int)
ok_panel_audit = panel_positions_audit >= 0
causal_positions_audit[ok_panel_audit] = causal_position[panel_positions_audit[ok_panel_audit]]
fit_rows_audit = np.flatnonzero(
    (causal_positions_audit >= 0) & np.isin(np.arange(len(sx)), valid_previous_audit)
)
drug_flags_audit = np.column_stack((
    train_states0["has_insulin"], train_states0["has_norepinephrine"]
))
innov_targets_audit = np.zeros((len(sx), 2))
innov_targets_audit[fit_rows_audit] = (
    dose_resid[causal_positions_audit[fit_rows_audit]]
    * drug_flags_audit[fit_rows_audit]
)
hidden_features_audit = post_hidden_audit.numpy()
folds_audit = train_states0["stay_id"] % 2
posterior_recovery_audit = {}
for alpha in (0.1, 1, 10, 100):
    prediction = np.zeros((len(fit_rows_audit), 2))
    for heldout in (0, 1):
        train_mask = folds_audit[fit_rows_audit] != heldout
        test_mask = ~train_mask
        xx = hidden_features_audit[fit_rows_audit]
        center = xx[train_mask].mean(0)
        spread = xx[train_mask].std(0).clip(1e-4)
        reg = Ridge(alpha=alpha).fit(
            (xx[train_mask] - center) / spread,
            innov_targets_audit[fit_rows_audit][train_mask]
        )
        prediction[test_mask] = reg.predict((xx[test_mask] - center) / spread)
    result = {"all": r2_score(innov_targets_audit[fit_rows_audit], prediction,
                             multioutput="raw_values").tolist()}
    for j, drug in enumerate(drug_codes):
        active = drug_flags_audit[fit_rows_audit, j]
        result[drug + "_active"] = r2_score(
            innov_targets_audit[fit_rows_audit[active], j], prediction[active, j])
    posterior_recovery_audit[alpha] = result
print("Posterior-feature innovation recovery:", posterior_recovery_audit)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 17: complete posterior and centered-signal audit\n"
            "- Single-start drug posterior reconstruction validated at RMS below 6e-6.\n"
            f"- END convention candidates: {ending_candidates_audit[:5]}\n"
            f"- Posterior innovation recovery: {posterior_recovery_audit}\n"
            "- Python runs: 17; forward passes: 4; no weights edited.\n")
```