Errors also occur on end-of-infusion bundles without new drugs, so measurement units alone cannot explain them. I’ll test whether the saved states actually use an encoded empty action or the latent prior on these bundles. Near-exact agreement with one alternative would resolve the discrepancy; otherwise I’ll return to atom encoding.

```python
print("=== EMPTY-SET AND PRIOR CONVENTIONS ===")
empty_e93 = encode_atom_set([], "encode")
empty_u93 = encode_atom_set([], "encode_action")
print("Empty observation RMS:", float(empty_e93.square().mean().sqrt()))
print("Empty action RMS:", float(empty_u93.square().mean().sqrt()))

tests93 = []
for e_mode93 in ["recorded", "zero", "encoded_empty"]:
    e93 = posterior_e_repair.clone()
    if e_mode93 == "zero":
        e93[~mask91] = 0
    elif e_mode93 == "encoded_empty":
        e93[~mask91] = empty_e93
    for u_mode93 in ["zero", "encoded_empty"]:
        u93 = posterior_u_repair.clone()
        u93[~mask91] = 0 if u_mode93 == "zero" else empty_u93
        z93 = rmlp(torch.cat([posterior_ctx_repair, e93, u93], 1), "z_post")[:, :32]
        er93 = (z93 - truth91).square().mean(1).sqrt().numpy()
        tests93.append({
            "e": e_mode93, "u": u_mode93,
            "no_action_mean": float(er93[~mask91].mean()),
            "no_action_max": float(er93[~mask91].max()),
            "action_mean": float(er93[mask91].mean())
        })
zprior93 = rmlp(posterior_ctx_repair, "z_prior")[:, :32]
erprior93 = (zprior93 - truth91).square().mean(1).sqrt().numpy()
tests93.append({
    "e": "prior_only", "u": "",
    "no_action_mean": float(erprior93[~mask91].mean()),
    "no_action_max": float(erprior93[~mask91].max()),
    "action_mean": float(erprior93[mask91].mean())
})
print(pd.DataFrame(tests93).sort_values("no_action_mean").to_string(index=False))
print("\nQuality embedding:", SHAPES.get("embed.kq.weight"))
print("Original/repair state arrays:", refx.shape, sx_repair.shape)
print("State array maximum difference:",
      float(np.max(np.abs(refx - np.asarray(sx_repair)))))
print("Current edits:", edited())
```
