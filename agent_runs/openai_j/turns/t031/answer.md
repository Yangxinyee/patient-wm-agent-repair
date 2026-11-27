Repeat predictions differ by 0.072 in scaled RMSE, consistent with about 0.051 noise per pass—too small to explain the 0.114 reconstruction error and positive bias. Stay IDs also match. I’ll rebuild the head using fresh metadata-derived inputs to eliminate stale notebook variables, then check register age and normalization placement.

```python
def reconstruct_reading_head(meta, state_array=SX, flow_factor=1.0,
                             age_before_bundle=False, norm_mode="query",
                             head_activation="gelu"):
    ix = meta.state_i.to_numpy(dtype=int)
    state = torch.as_tensor(state_array[ix], dtype=torch.float32)
    out_local = state[:, :192]
    h_local = state[:, 192:960].reshape(-1, 4, 192)
    dt_local = torch.as_tensor(meta.dt.to_numpy(), dtype=torch.float32)
    flow_local = h_local * torch.exp(
        -flow_factor * dt_local[:, None, None]
        * torch.exp(wt("core.flow.a"))[None, None, :]
    )
    gap_local = mlp(torch.log1p(dt_local)[:, None]/10, "gap", "gelu")
    context = out_local + ln(flow_local.mean(1), "core.out_norm") + gap_local
    prior_local = mlp(context, "z_prior", "gelu")
    view_local = context + TF.linear(
        prior_local[:, :32], wt("z_head.weight"), wt("z_head.bias")
    )
    code_local = torch.as_tensor(meta.code.to_numpy(), dtype=torch.long)
    bin_local = torch.as_tensor(meta.last_bin.to_numpy(), dtype=torch.long)
    numbers = torch.zeros(len(meta), 8)
    numbers[:, 0] = torch.as_tensor(meta.has_last.to_numpy(), dtype=torch.float32)
    ages = meta.age_s.to_numpy().copy()
    if age_before_bundle:
        ages = np.maximum(ages - meta.dt.to_numpy(), 0)
    numbers[:, 1] = torch.as_tensor(np.log1p(ages)/10, dtype=torch.float32)
    numbers[:, 2] = torch.as_tensor(
        np.clip(meta.zlast.to_numpy(), -10, 10), dtype=torch.float32
    )
    numbers[:, 3] = numbers[:, 0]
    register_local = (
        wt("embed.code.weight")[code_local]
        + TF.linear(numbers, wt("last_numeric.weight"), wt("last_numeric.bias"))
        + wt("embed.bins.value.weight")[bin_local]
        + wt("embed.bins.rate.weight")[0]
        + wt("last_status.weight")[0]
    )
    query_input = view_local + wt("type_emb.weight")[1]
    if norm_mode == "query":
        inputs = ln(query_input, "query_norm") + register_local
    elif norm_mode == "combined":
        inputs = ln(query_input + register_local, "query_norm")
    elif norm_mode == "none":
        inputs = query_input + register_local
    else:
        raise ValueError(norm_mode)
    features = activation(TF.linear(
        inputs, wt("attr_heads.value_bin.0.weight"),
        wt("attr_heads.value_bin.0.bias")
    ), head_activation)
    logits_local = (
        TF.linear(features, wt("attr_heads.value_bin.2.weight"),
                  wt("attr_heads.value_bin.2.bias"))
        + wt("embed_transition.value_bin.weight")[code_local*17 + bin_local]
    )
    return {
        "state": state, "dt": dt_local, "context": context, "view": view_local,
        "register": register_local, "numbers": numbers,
        "inputs": inputs, "hidden": features, "probs": logits_local.softmax(-1),
        "logits": logits_local, "code": code_local, "last_bin": bin_local,
    }

fresh_head = reconstruct_reading_head(hm)
print("=== FRESH VS EXISTING INPUTS ===")
for label, fresh, old in [
    ("saved state", fresh_head["state"], sx),
    ("dt", fresh_head["dt"], dt),
    ("context", fresh_head["context"], ctx_default),
    ("view", fresh_head["view"], view_default),
    ("register", fresh_head["register"], reg_default),
    ("numbers", fresh_head["numbers"], nums),
]:
    print(label, tensor_rmse(fresh, old))

fresh_checks = []
for label, options in [
    ("fresh_documented", {}),
    ("age_at_preceding_bundle", {"age_before_bundle": True}),
    ("normalize_combined", {"norm_mode": "combined"}),
    ("no_query_normalization", {"norm_mode": "none"}),
    ("half_time_flow", {"flow_factor": 0.5}),
    ("double_time_flow", {"flow_factor": 2.0}),
    ("silu_value_head_only", {"head_activation": "silu"}),
]:
    output = reconstruct_reading_head(hm, **options)
    mu_local = (output["probs"] * centres_t).sum(-1).numpy()
    nll_local = -output["probs"][torch.arange(len(hm)), target_t].clamp_min(1e-12).log().numpy()
    fresh_checks.append({
        "variant": label,
        "scaled_mean_rmse": np.sqrt(np.mean(((mu_local-target_mu)/scale_mu)**2)),
        "scaled_bias": np.mean((mu_local-target_mu)/scale_mu),
        "nll_rmse": np.sqrt(np.mean((nll_local-hm.original_nll.to_numpy())**2)),
    })
print("\n=== FRESH CHECKS ===")
print(pd.DataFrame(fresh_checks).sort_values("scaled_mean_rmse").to_string(index=False))
print("\ncentre-table structure:", type(VOCAB["value_bin_centres"]).__name__)
print(str(VOCAB["value_bin_centres"])[:650])
print("first reconstruction centre row:", centres_t[0].numpy())

with open("/work/notes.md", "a") as notebook:
    notebook.write(
        "\n- Run 25: repeated factual predictions differ by scaled mean RMSE "
        "0.07198 and NLL RMSE 0.24980, implying single-pass sampling noise "
        "~0.051. Reconstruction has larger systematic positive bias. "
        "All matched state IDs are correct. Insulin kernel bootstrap is very "
        "weak (rho often reaches bounds); NE kernel is substantially more stable. "
        "Treat long-horizon insulin targets as uncertain. No edits; five forward passes.\n"
    )
```