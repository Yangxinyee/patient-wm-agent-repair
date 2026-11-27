Fresh and stored inputs agree, and fresh singleton drug starts reproduce the posterior to about \(10^{-5}\). The mismatch is specific to non-action bundles. I’ll test whether those bundles use an encoded empty action set rather than a zero vector. Exact agreement would confirm that convention.

```python
empty_action_encoded = recon_norm(
    recon_mlp(recon_lin(torch.zeros((1, 1)), "encode_action.size"),
              "encode_action.mlp2"),
    "encode_action.norm"
)
print("empty observation norm:", float(empty_e.norm()),
      "empty action norm:", float(empty_action_encoded.norm()))
empty_action_test = {}
for code in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    ii = []
    for i in recon_valid_rows:
        if len(recon_obs_lists[i]) == 1 and not recon_action_lists[i]:
            row = state_df.iloc[i]
            sub = tune[(tune.stay_id == row.stay_id) & (np.abs(tune.hour-row.hour) < 1e-5)]
            if len(sub) == 1 and sub.iloc[0].event == code:
                ii.append(int(i))
        if len(ii) >= 60:
            break
    atoms = torch.stack([recon_obs_lists[i][0] for i in ii])
    ee3 = single_encode_batch(atoms, "encode")
    ctx3 = recon_ctx(ii, gap_style="log")
    target3 = torch.as_tensor(S0["x"][ii, 960:992])
    results3 = {}
    for label, u3 in [
        ("zero", torch.zeros_like(ee3)),
        ("encoded_empty", empty_action_encoded.repeat(len(ii), 1)),
        ("observation_encoder_empty", empty_e.repeat(len(ii), 1)),
    ]:
        pred3 = recon_mlp(torch.cat([ctx3, ee3, u3], dim=1), POST_PREFIX)[:, :32]
        results3[label] = float(((pred3-target3)**2).mean().sqrt())
    empty_action_test[code] = results3
    print(code, results3)
print("\nOther categorical embedding norms (rows 0 and 1):")
for name in NAMES:
    if name.startswith(("embed.categorical.", "embed.text")) and name.endswith("weight"):
        print(name, [round(float(x.norm()), 4) for x in W0[name][:2]])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 27: empty-action convention test\n"
            + str(empty_action_test) +
            "\nNo edits. Python runs: 27; forward passes: 3.\n")
```
