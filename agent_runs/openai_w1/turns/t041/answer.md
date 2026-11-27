The posterior reconstruction error is still too large to trust. One specific omission may explain it: an empty observation bundle can still produce a nonzero encoder vector. I’ll test that, END routing, and posterior layout before editing. Near-exact agreement would validate this route; persistent error would make it unsafe.

```python
def r34_activation(x, kind):
    return torch.tanh(x) if kind == "tanh" else r33_act(x, kind)

def r34_mlp(x, prefix, kind):
    return r33_linear(r34_activation(r33_linear(x, prefix + ".0"), kind),
                      prefix + ".2")

def r34_set_encode(atoms, owners, mask, prefix, kind, empty_zero):
    counts = torch.zeros(len(r33_sample))
    counts.index_add_(0, owners[mask], torch.ones(int(mask.sum())))
    sums = torch.zeros(len(r33_sample), 192)
    if bool(mask.any()):
        sums.index_add_(0, owners[mask],
                        r34_mlp(atoms[mask], prefix + ".mlp1", kind))
    encoded = r33_ln(r34_mlp(
        sums/counts.clamp_min(1).sqrt()[:, None]
        + r33_linear(counts.log1p()[:, None], prefix + ".size"),
        prefix + ".mlp2", kind), prefix + ".norm")
    if empty_zero:
        encoded[counts == 0] = 0
    return encoded

r34_contexts = {}
for r34_ga, r34_base in itertools.product(
        ["silu", "gelu", "relu"], ["out_plus_flow", "flow_only", "out_only"]):
    r34_gap = r34_mlp((r33_dt.clamp_min(1).log()*.1)[:, None], "gap", r34_ga)
    r34_contexts[(r34_ga, r34_base)] = {
        "out_plus_flow": r33_previous[:, :192] + r33_flow_out,
        "flow_only": r33_flow_out,
        "out_only": r33_previous[:, :192]
    }[r34_base] + r34_gap

r34_results = []
r34_best_error = float("inf")
for r34_end, r34_side in itertools.product(["none", "copy"], ["left", "right"]):
    r34_atoms, r34_owners = r33_action_embedding(r34_end, (), True, r34_side)
    r34_is_start = []
    for r34_i in r33_sample:
        r34_key = (int(r33_ids[r34_i]), round(float(r33_hours[r34_i]), 4))
        r34_is_start.extend([True]*len(r33_starts.get(r34_key, [])))
        r34_is_start.extend([False]*len(r33_ends.get(r34_key, [])))
    r34_is_start = torch.tensor(r34_is_start)
    for r34_ea in ["silu", "gelu", "relu"]:
        for r34_route in ["all_action", "starts_action_ends_observation"]:
            r34_u_mask = (torch.ones(len(r34_owners), dtype=torch.bool)
                          if r34_route == "all_action" else r34_is_start)
            r34_e_mask = (~r34_is_start if r34_route != "all_action"
                          else torch.zeros(len(r34_owners), dtype=torch.bool))
            r34_u = r34_set_encode(r34_atoms, r34_owners, r34_u_mask,
                                   "encode_action", r34_ea, True)
            for r34_empty_zero in [False, True]:
                r34_e = r34_set_encode(r34_atoms, r34_owners, r34_e_mask,
                                       "encode", r34_ea, r34_empty_zero)
                r34_ue = F.linear(
                    torch.cat([r34_e, r34_u], 1),
                    r33_weights["z_post.0.weight"][:, 192:])
                for r34_ck, r34_ctx in r34_contexts.items():
                    r34_pre = F.linear(
                        r34_ctx, r33_weights["z_post.0.weight"][:, :192],
                        r33_weights["z_post.0.bias"]) + r34_ue
                    for r34_za in ["silu", "gelu", "relu", "tanh"]:
                        r34_hidden = r34_activation(r34_pre, r34_za)
                        r34_raw = r33_linear(r34_hidden, "z_post.2")
                        r34_prior = r34_mlp(r34_ctx, "z_prior", r34_za)
                        for r34_layout in ["first", "last", "even", "odd"]:
                            r34_columns = {
                                "first": slice(0, 32), "last": slice(32, 64),
                                "even": slice(0, 64, 2), "odd": slice(1, 64, 2)
                            }[r34_layout]
                            r34_mu = r34_raw[:, r34_columns]
                            r34_pm = r34_prior[:, r34_columns]
                            for r34_mode in ["direct", "plus_prior", "minus_prior"]:
                                r34_pred = {
                                    "direct": r34_mu,
                                    "plus_prior": r34_mu+r34_pm,
                                    "minus_prior": r34_mu-r34_pm
                                }[r34_mode]
                                r34_error = float(
                                    (r34_pred-r33_target).square().mean().sqrt())
                                r34_cfg = (r34_end, r34_side, r34_ea, r34_route,
                                           r34_empty_zero, *r34_ck, r34_za,
                                           r34_layout, r34_mode)
                                r34_results.append((r34_error, r34_cfg))
                                if r34_error < r34_best_error:
                                    r34_best_error = r34_error
                                    r34_best_cfg = r34_cfg
                                    r34_best_pred = r34_pred.clone()
                                    r34_best_hidden = r34_hidden.clone()
                                    r34_best_ctx = r34_ctx.clone()
                                    r34_best_e = r34_e.clone()
                                    r34_best_u = r34_u.clone()

r34_results.sort(key=lambda x: x[0])
print("Best posterior reconstructions:")
for r34_row in r34_results[:12]:
    print(r34_row)
print("Best per-latent RMSE:",
      np.round((r34_best_pred-r33_target).square().mean(0).sqrt().numpy(), 4))
print("Predicted/observed latent SD:",
      float(r34_best_pred.std()), float(r33_target.std()))

with open("/work/notes.md", "a") as r34_file:
    r34_file.write(
        "\n## Run 34\n"
        "Tested empty observation-encoder output, END routing, posterior layouts "
        "and prior residual parameterization against exported latent means.\n"
        f"Best: {r34_results[:3]}\n"
        "Counts: 34 Python runs, 6 forward passes. No edits.\n")
```
