The lookup keys are `(stay_id, hour)` pairs; the retained bundle list already provides the correct state indices. I’ll now measure how well action vectors preserve rate information and how much changing rate moves the posterior mean. I’ll also test the simple possibility that END events are omitted from both encoders. Strong rate decoding would help us construct a correction even if END encoding remains unresolved.

```python
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

print("Kinds:", VOCAB["atom_kinds"], VOCAB["event_kinds"])

action_cases50 = [(i, starts, ends) for i, starts, ends in latent_bundles if starts]
action_indices50 = np.array([i for i, _, _ in action_cases50])
single_clean50 = [(i, starts[0]) for i, starts, ends in action_cases50
                  if len(starts) == 1 and not ends]
print("Action cases:", len(action_cases50), "clean single starts:", len(single_clean50))

def action_vector50(rows):
    return latent_action_encode(torch.stack([latent_start_atom(r) for r in rows])).reshape(192)

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
    print(drug, "n=", len(rows), "dose-direction norm=", np.linalg.norm(direction),
          "mean latent change norm=", np.linalg.norm(dz, axis=1).mean())
    globals().setdefault("dose_directions50", {})[drug] = direction
```
