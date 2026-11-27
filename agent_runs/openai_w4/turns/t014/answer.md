Dose innovations are only weakly recoverable: the best held-out scores are roughly 17% for insulin and 11% for norepinephrine. A readout-only edit could therefore amplify a poor proxy. I’ll first reconstruct the head arithmetic from exported states; matching the forward tool’s predictions would make later weight experiments interpretable.

```python
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
W_repair = {
    name: torch.as_tensor(np.asarray(load_original(name)).copy(), dtype=torch.float32)
    for name in NAMES
}
def linear_repair(x, prefix):
    return F.linear(x, W_repair[prefix + ".weight"],
                    W_repair.get(prefix + ".bias"))
def norm_repair(x, prefix):
    return F.layer_norm(x, (192,), W_repair[prefix + ".weight"],
                        W_repair[prefix + ".bias"], eps=1e-5)
def act_repair(x, kind):
    if kind == "gelu":
        return F.gelu(x)
    if kind == "silu":
        return F.silu(x)
    return F.relu(x)
def mlp_repair(x, prefix, kind):
    return linear_repair(act_repair(linear_repair(x, prefix + ".0"), kind),
                         prefix + ".2")

print("=== PADDING/CATEGORICAL NORMS ===")
for name in NAMES:
    if name.startswith("embed.categorical.") or name in (
        "embed.kind.weight", "embed.ekind.weight",
        "embed.bins.value.weight", "embed.bins.rate.weight"
    ):
        print(name, np.round(np.linalg.norm(W_repair[name].numpy(), axis=1), 3).tolist())
print("flow.a range:", float(W_repair["core.flow.a"].min()),
      float(W_repair["core.flow.a"].max()))

selected_records = tuning[tuning.stay_id.isin(tuning_ids_repair)].copy()
csv_bundles = selected_records.groupby(["stay_id", "hour"]).size()
print("CSV unique bundles:", len(csv_bundles),
      "exported bundles:", len(state_o["hour"]))
print("Action n_atoms distribution:", np.unique(
    state_o["n_atoms"][state_o["has_insulin"] | state_o["has_norepinephrine"]],
    return_counts=True
))

numeric_lookup_repair = {
    (x["code"], x["attribute"]): x for x in VOCAB["numeric"]
}
code_lookup_repair = {x["name"]: x["id"] for x in VOCAB["codes"]}
state_banks_repair = {
    int(sid): np.flatnonzero(state_o["stay_id"] == sid)
    for sid in tuning_ids_repair
}
reading_banks_repair = {}
for (sid, event), group in selected_records[
    selected_records.value.notna()
].groupby(["stay_id", "event"]):
    group = group.sort_values("hour")
    reading_banks_repair[(int(sid), event)] = (
        group.hour.to_numpy(), group.value.to_numpy()
    )

n_fact_repair = len(baseline_fact)
fact_prev_repair = np.empty(n_fact_repair, dtype=int)
fact_dt_repair = np.empty(n_fact_repair, dtype=float)
fact_last_z_repair = np.zeros(n_fact_repair, dtype=float)
fact_age_repair = np.zeros(n_fact_repair, dtype=float)
fact_has_repair = np.zeros(n_fact_repair, dtype=float)
fact_bin_repair = np.zeros(n_fact_repair, dtype=int)
fact_code_repair = np.empty(n_fact_repair, dtype=int)
fact_observed_bin_repair = np.empty(n_fact_repair, dtype=int)
fact_centres_repair = np.zeros((n_fact_repair, 17), dtype=np.float32)

for j, row in enumerate(baseline_fact.itertuples(index=False)):
    indices = state_banks_repair[int(row.stay_id)]
    position = np.searchsorted(state_o["hour"][indices], row.hour - 1e-6) - 1
    if position < 0:
        raise ValueError("Missing preceding state")
    prev = indices[position]
    fact_prev_repair[j] = prev
    fact_dt_repair[j] = (row.hour - state_o["hour"][prev]) * 3600
    info = numeric_lookup_repair[(row.event, "value")]
    edges = np.asarray(info["fine_bin_edges"])
    fact_code_repair[j] = code_lookup_repair[row.event]
    fact_observed_bin_repair[j] = 1 + np.searchsorted(edges, row.observed, side="left")
    fact_centres_repair[j, 1:] = VOCAB["value_bin_centres"][row.event]
    times, values = reading_banks_repair[(int(row.stay_id), row.event)]
    lp = np.searchsorted(times, row.hour - 1e-6) - 1
    if lp >= 0:
        fact_has_repair[j] = 1
        fact_last_z_repair[j] = np.clip(
            (values[lp] - info["median"]) / info["iqr"], -10, 10
        )
        fact_age_repair[j] = np.log1p((row.hour - times[lp]) * 3600) / 10
        fact_bin_repair[j] = 1 + np.searchsorted(edges, values[lp], side="left")

fact_state_repair = torch.from_numpy(state_o["x"][fact_prev_repair])
flow_h_repair = fact_state_repair[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -torch.tensor(fact_dt_repair, dtype=torch.float32)[:, None, None]
    * torch.exp(W_repair["core.flow.a"])[None, None, :]
)
context_base_repair = (
    fact_state_repair[:, :192]
    + norm_repair(flow_h_repair.mean(dim=1), "core.out_norm")
)
gap_input_repair = torch.tensor(
    np.log(np.maximum(fact_dt_repair, 1.0)) / 10, dtype=torch.float32
)[:, None]
code_t_repair = torch.tensor(fact_code_repair)
bin_t_repair = torch.tensor(fact_bin_repair)
h, a, z = fact_has_repair, fact_age_repair, fact_last_z_repair
zero = np.zeros(n_fact_repair)
last_patterns_repair = {
    "has_age_numeric": np.column_stack([h, a, z, h, zero, zero, zero, zero]),
    "numeric_has_age": np.column_stack([z, h, zero, zero, zero, zero, h, a]),
    "four_then_zero": np.column_stack([h, a, z, zero, zero, zero, zero, zero]),
    "has_age_zrate_zdose_flags": np.column_stack([h, a, z, zero, zero, h, zero, zero]),
}
reference_mean_repair = baseline_fact.predicted_mean.to_numpy()
reference_nll_repair = baseline_fact.nll.to_numpy()
head_candidates_repair = []
head_predictions_repair = {}

for activation in ("gelu", "silu", "relu"):
    context = context_base_repair + mlp_repair(gap_input_repair, "gap", activation)
    prior_mean = mlp_repair(context, "z_prior", activation)[:, :32]
    view = context + linear_repair(prior_mean, "z_head")
    for type_index in (0, 1):
        q = norm_repair(view + W_repair["type_emb.weight"][type_index], "query_norm")
        base = (
            q + W_repair["embed.code.weight"][code_t_repair]
            + W_repair["embed.bins.value.weight"][bin_t_repair]
            + W_repair["embed.bins.rate.weight"][0]
            + W_repair["last_status.weight"][0]
        )
        for pattern, last_features in last_patterns_repair.items():
            with_code = base + linear_repair(
                torch.tensor(last_features, dtype=torch.float32), "last_numeric"
            )
            hidden = act_repair(linear_repair(with_code, "attr_heads.value_bin.0"),
                                activation)
            logits = linear_repair(hidden, "attr_heads.value_bin.2")
            logits += W_repair["embed_transition.value_bin.weight"][
                code_t_repair * 17 + bin_t_repair
            ]
            prob = F.softmax(logits, dim=-1).numpy()
            pred = np.sum(prob * fact_centres_repair, axis=1)
            nll = -np.log(np.maximum(
                prob[np.arange(n_fact_repair), fact_observed_bin_repair], 1e-30
            ))
            key = (activation, type_index, pattern)
            head_predictions_repair[key] = (pred, nll)
            head_candidates_repair.append({
                "activation": activation, "type": type_index, "last_pattern": pattern,
                "mean_RMSE": np.sqrt(np.mean((pred - reference_mean_repair)**2)),
                "nll_RMSE": np.sqrt(np.mean((nll - reference_nll_repair)**2)),
                "mean_nll": nll.mean(),
            })
head_candidate_table_repair = pd.DataFrame(head_candidates_repair).sort_values("nll_RMSE")
print("\n=== HEAD RECONSTRUCTION CANDIDATES ===")
print(head_candidate_table_repair.head(12).round(5).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 10\n"
        "Dose innovations decode weakly from original states (best approximately "
        "0.17 insulin R2 and 0.11 norepinephrine R2). "
        "Readout-only repair is not yet justified; investigating action-pathway repair.\n"
        "Implemented candidate CPU head reconstructions using exported states; "
        "tested activation, measurement type index and last-numeric ordering "
        "against original forward predictions.\n"
        + head_candidate_table_repair.head(8).round(5).to_string(index=False)
        + "\nNo edits.\n"
    )
```