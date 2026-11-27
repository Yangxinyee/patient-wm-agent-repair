The timing tests did not produce a setting that fixes readings while preserving the exact drug-start match. I’ll now check omitted embedding fields and encoder details. This is still a reconstruction diagnostic—not a weight edit—and only a near-exact match across sensors will count as confirmation.

```python
print("=== POSSIBLY OMITTED EMBEDDINGS ===")
for n in ("embed.kq.weight", "embed.text.weight", "embed.categorical.rate_unit.weight",
          "embed.bins.value.weight", "embed.bins.dose.weight", "embed.bins.rate.weight"):
    ww0 = W[n]
    print(n, tuple(ww0.shape), "row-0 norm", float(ww0[0].norm()),
          "first row norms", [float(rr0.norm()) for rr0 in ww0[:5]])
print("numeric bias norm:", float(W["embed.numeric.bias"].norm()))
for event0 in ("glucose_finger", "map_cuff", "insulin"):
    attr0 = "rate" if event0 == "insulin" else "value"
    print(event0, "edges:", numeric_tables[(event0, attr0)]["fine_bin_edges"])

obs_diagnostics = {}
with torch.no_grad():
    for event0 in ("map_cuff", "glucose_finger", "glucose_lab"):
        idx0 = np.array(isolated_by_event[event0][:24])
        p0 = np.array([pos_by_idx[int(i0)] for i0 in idx0])
        atom0 = torch.stack([make_atom(bundle_atom_info[i0][0][0], False, "preserve")
                             for i0 in idx0])
        tests0 = []
        for kq0 in range(W["embed.kq.weight"].shape[0]):
            for text0 in range(min(W["embed.text.weight"].shape[0], 3)):
                aa0 = atom0 + W["embed.kq.weight"][kq0] - W["embed.kq.weight"][0]
                if text0:
                    aa0 = aa0 + W["embed.text.weight"][text0]
                ee0 = encode_one_set(aa0, "encode")
                zz0 = mlp(torch.cat([ctx_all[p0], ee0, torch.zeros_like(ee0)], 1),
                          "z_post", "gelu")[:, :32]
                tests0.append((float(((zz0-sx[idx0, -32:])**2).mean()),
                               "extra_fields", kq0, text0))
        for act1, act2, divisor, size_input in itertools.product(
                ("gelu", "silu", "relu"), ("gelu", "silu", "relu"),
                (1., np.sqrt(2), 2.), (np.log(2), 1.)):
            hid0 = mlp(atom0, "encode.mlp1", act1) / divisor
            sz0 = torch.full((len(idx0), 1), size_input, dtype=torch.float32)
            ee0 = layer_norm(mlp(hid0 + linear(sz0, "encode.size"),
                                "encode.mlp2", act2), "encode.norm")
            zz0 = mlp(torch.cat([ctx_all[p0], ee0, torch.zeros_like(ee0)], 1),
                      "z_post", "gelu")[:, :32]
            tests0.append((float(((zz0-sx[idx0, -32:])**2).mean()),
                           "encoder", act1, act2, divisor, size_input))
        obs_diagnostics[event0] = sorted(tests0, key=lambda t0: t0[0])[:5]
        print(event0, obs_diagnostics[event0])
```