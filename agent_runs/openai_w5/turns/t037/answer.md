Unit mappings alone did not reproduce the latent states closely enough, so the local encoder is not validated yet. The learned quality embedding is another possible missing input. I’ll test it, the empty-bundle convention, and the activation choice. Near-zero latent error would confirm the reconstruction.

```python
for key in ("embed.bins.value.weight", "embed.bins.dose.weight", "embed.bins.rate.weight"):
    print(key, "absent-bin norm", float(torch.linalg.vector_norm(W0[key][0])))

action_reconstruction_trials = {}
for drug, probe in action_probe_data.items():
    n = len(probe["rows"])
    trials = []
    for activation in ("gelu", "silu", "relu"):
        zero_size = torch.zeros(n, 1)
        empty_e = norm0(mlp0(lin0(zero_size, "encode.size"),
                             "encode.mlp2", activation), "encode.norm")
        for unit, rate_unit, quality in product(range(6), range(4), range(-1, 4)):
            atoms = atom_action_local(drug, probe["rates"], rate_unit=rate_unit, unit=unit)
            if quality >= 0:
                atoms = atoms + W0["embed.kq.weight"][quality]
            a1 = mlp0(atoms, "encode_action.mlp1", activation)
            action_u = norm0(
                mlp0(a1 + lin0(torch.full((n, 1), np.log(2)), "encode_action.size"),
                     "encode_action.mlp2", activation), "encode_action.norm")
            for empty_mode in ("zero", "encoded"):
                e = torch.zeros_like(action_u) if empty_mode == "zero" else empty_e
                zp = mlp0(torch.cat([probe["ctx"], e, action_u], dim=1),
                          "z_post", activation)[:, :32]
                target_z = torch.from_numpy(state4["x"][probe["rows"], 960:])
                error = float((zp-target_z).square().mean().sqrt())
                trials.append((error, activation, unit, rate_unit, quality, empty_mode))
    trials.sort()
    action_reconstruction_trials[drug] = trials
    print(drug, "best:", trials[:8])

# Check the state layout independently of input reconstruction.
sx = torch.from_numpy(state4["x"][:500])
out_from_h = norm0(sx[:, 192:960].reshape(-1, 4, 192).mean(1), "core.out_norm")
print("out-layout RMSE:", float((out_from_h-sx[:, :192]).square().mean().sqrt()))
```
