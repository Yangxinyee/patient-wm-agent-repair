The paired-sensor check supports an immediate insulin effect, but is too weak to identify norepinephrine’s effect; the latter relies on stronger history-control assumptions. I’ll now change doses inside the action encoder while holding factual context fixed. This will test whether the posterior features offer a usable dose-response direction without large factual changes.

```python
import torch.nn.functional as TF
torch.set_num_threads(8)

print("=== STATE OBJECTS ===")
for name in ["repair_sf", "repair_states", "repair_context", "repair_truth_z",
             "repair_valid", "repair_has_action", "repair_has_end"]:
    obj = globals()[name]
    print(name, type(obj).__name__, getattr(obj, "shape", None))
    if isinstance(obj, pd.DataFrame):
        print("columns:", obj.columns.tolist(), "first:", obj.head(1).to_dict("records"))
    elif isinstance(obj, dict) or hasattr(obj, "files"):
        print("keys:", list(obj.keys())[:12])
print("Panel hours:", HOURS, "first stay ids:", gs[:3])

post_av0 = repair_atom_vectors(False, False)
post_action_mask = np.asarray(repair_is_action, dtype=bool)
post_e0 = repair_set_encode(post_av0, ~post_action_mask, "encode", False)
post_u0 = repair_set_encode(post_av0, post_action_mask, "encode_action", True)
post_ctx0 = torch.as_tensor(repair_context, dtype=torch.float32)
post_h0 = TF.gelu(TF.linear(torch.cat([post_ctx0, post_e0, post_u0], 1),
                           wt("z_post.0.weight"), wt("z_post.0.bias")))
post_z0 = TF.linear(post_h0, wt("z_post.2.weight"), wt("z_post.2.bias"))
post_truth = torch.as_tensor(repair_truth_z, dtype=torch.float32)
exact_mask = np.asarray(repair_has_action) & ~np.asarray(repair_has_end)
print("Posterior mean layout RMSE:",
      "first half", tensor_rmse(post_z0[exact_mask, :32], post_truth[exact_mask]),
      "interleaved", tensor_rmse(post_z0[exact_mask, ::2], post_truth[exact_mask]))

post_cf = {}
post_rates = np.zeros((repair_n, 2), dtype=np.float32)
post_atom_bundle = repair_atoms.state_index.to_numpy(dtype=int)
for dj, drug in enumerate(["insulin", "norepinephrine"]):
    am = ((repair_atoms.event.to_numpy() == drug) & post_action_mask
          & ~repair_atoms.is_end.to_numpy(dtype=bool))
    rows = np.flatnonzero(am)
    np.add.at(post_rates[:, dj], post_atom_bundle[rows],
              repair_atoms.iloc[rows].rate.to_numpy(dtype=np.float32))
    for factor in [0.0, 0.5, 1.5, 2.0, 3.0]:
        av = post_av0.clone()
        for attr, number_column in [("dose", 2), ("rate", 4)]:
            norm = repair_norms[(drug, attr)]
            old = repair_atoms.iloc[rows][attr].to_numpy(dtype=np.float32)
            new = factor * old
            old_z = np.clip((old - norm["median"]) / norm["iqr"], -10, 10)
            new_z = np.clip((new - norm["median"]) / norm["iqr"], -10, 10)
            old_b = 1 + np.searchsorted(norm["fine_bin_edges"], old, side="left")
            new_b = 1 + np.searchsorted(norm["fine_bin_edges"], new, side="left")
            av[rows] += (torch.as_tensor(new_z - old_z)[:, None]
                         * wt("embed.numeric.weight")[:, number_column][None, :])
            av[rows] += (wt("embed.bins." + attr + ".weight")[new_b]
                         - wt("embed.bins." + attr + ".weight")[old_b])
        u_cf = repair_set_encode(av, post_action_mask, "encode_action", True)
        h_cf = TF.gelu(TF.linear(torch.cat([post_ctx0, post_e0, u_cf], 1),
                                wt("z_post.0.weight"), wt("z_post.0.bias")))
        post_cf[(drug, factor)] = h_cf.numpy()
    action_rows = post_rates[:, dj] > 0
    dh = post_cf[(drug, 2.0)][action_rows] - post_h0.numpy()[action_rows]
    print(drug, "active bundles", int(action_rows.sum()),
          "mean rate", float(post_rates[action_rows, dj].mean()),
          "x2 hidden-change RMS", float(np.sqrt(np.mean(dh**2))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 57: posterior counterfactual features\n"
            "- Held factual context fixed and computed action-encoder/posterior hidden features for factors 0, .5, 1.5, 2, 3.\n"
            "- Next test: a cross-stay regression with low factual activation and dose-proportional counterfactual differences.\n"
            "- No weight edits; nine forward passes used.\n")
```
