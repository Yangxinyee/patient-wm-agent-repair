There are 1,191 action bundles and 110 clean single-drug starts. The calculation stopped because the encoder helper expects a list, not a stacked tensor, so it yielded no sensitivity result. I’ll fix that call and rerun the checks; the important tests are held-out rate decoding and reproducible posterior movement when rate changes.

```python
torch.set_num_threads(8)

def action_vector50(rows):
    return latent_action_encode([latent_start_atom(r) for r in rows]).reshape(192)

action_context50 = latent_context(action_indices50)
action_u50 = torch.stack([action_vector50(starts) for _, starts, _ in action_cases50])
action_hidden50 = latent_hidden(action_context50, action_u50)
print("Feature shapes:", action_context50.shape, action_u50.shape, action_hidden50.shape)

clean_indices50 = np.array([i for i, _ in single_clean50])
clean_ctx50 = latent_context(clean_indices50)
clean_u50 = torch.stack([action_vector50([r]) for _, r in single_clean50])
clean_z50 = latent_post_mean(
    clean_ctx50, latent_empty_e.expand(len(clean_indices50), -1), clean_u50)
print("Clean start posterior RMS:",
      float(torch.sqrt(torch.mean((clean_z50 -
             torch.from_numpy(latent_state[clean_indices50, -32:]))**2))))

end_test_indices50 = np.asarray(end_idx41, dtype=int)
end_test_ctx50 = latent_context(end_test_indices50)
end_empty_z50 = latent_post_mean(
    end_test_ctx50, latent_empty_e.expand(len(end_test_indices50), -1),
    torch.zeros(len(end_test_indices50), 192))
print("END with empty observation, zero action RMS:",
      float(torch.sqrt(torch.mean((end_empty_z50 -
             torch.from_numpy(latent_state[end_test_indices50, -32:]))**2))))

rates50 = np.array([[sum(float(r["rate"]) for r in starts if r["event"] == drug)
                    for drug in DRUGS] for _, starts, _ in action_cases50])
ids50 = latent_sid[action_indices50]
train50 = np.isin(ids50, np.unique(ids50)[:65])
table50 = []
for j, drug in enumerate(DRUGS):
    for features_name, features in [("action_vector", action_u50.numpy()),
                                   ("context", action_context50.numpy()),
                                   ("posterior_hidden", action_hidden50.numpy())]:
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            reg = Ridge(alpha=alpha).fit(features[train50], rates50[train50, j])
            pred = reg.predict(features[~train50])
            table50.append([drug, features_name, alpha,
                            r2_score(rates50[~train50, j], pred),
                            np.sqrt(np.mean((pred-rates50[~train50, j])**2))])
print(pd.DataFrame(table50, columns=["drug", "features", "alpha", "heldout_R2", "RMSE"])
      .round(4).to_string(index=False))

dose_directions50 = {}
print("\nOriginal posterior sensitivity to doubling, clean starts:")
for drug in DRUGS:
    rows = [(i, r) for i, r in single_clean50 if r["event"] == drug]
    ix = np.array([i for i, _ in rows])
    cx = latent_context(ix)
    u1 = torch.stack([action_vector50([r]) for _, r in rows])
    changed = [{**r, "rate": 2*float(r["rate"]), "dose": 2*float(r["dose"])}
               for _, r in rows]
    u2 = torch.stack([action_vector50([r]) for r in changed])
    e = latent_empty_e.expand(len(rows), -1)
    dz = (latent_post_mean(cx, e, u2)-latent_post_mean(cx, e, u1)).numpy()
    dr = np.array([float(r["rate"]) for _, r in rows])
    direction = (dr @ dz)/(dr @ dr)
    dose_directions50[drug] = direction
    print(drug, "n=", len(rows), "direction norm=", round(np.linalg.norm(direction), 6),
          "mean latent change norm=", round(np.linalg.norm(dz, axis=1).mean(), 6))

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Python 51; forward 6 — dose representation checks\n"
             "- Corrected action encoder helper input (list of tensors).\n"
             "- Testing action/context rate decoding and original posterior dose derivatives. No edits.\n")
```
