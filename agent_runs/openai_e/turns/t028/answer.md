The absent-attribute terms are zero, so they do not explain the discrepancy. I’ll test event-kind assignments and the observation encoder’s activations, using readings away from bin edges. An exact match would resolve the encoder; otherwise I’ll revisit how bundle inputs are assembled.

```python
print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])
print("numeric note:", VOCAB.get("numeric_note"))
encoder_search = {}
for code in ["glucose_finger", "map_arterial", "map_cuff"]:
    ii = []
    edges0 = np.asarray(NUM_SPEC[(code, "value")]["fine_bin_edges"])
    for i in recon_valid_rows:
        if len(recon_obs_lists[i]) == 1 and not recon_action_lists[i]:
            row = state_df.iloc[i]
            sub = tune[(tune.stay_id == row.stay_id) & (np.abs(tune.hour-row.hour) < 1e-5)]
            if len(sub) == 1 and sub.iloc[0].event == code:
                if not np.any(edges0 == sub.iloc[0].value):
                    ii.append(int(i))
        if len(ii) >= 40:
            break
    aa0 = torch.stack([recon_obs_lists[i][0] for i in ii])
    target2 = torch.as_tensor(S0["x"][ii, 960:992])
    combinations2 = list(itertools.product(range(6), range(5)))
    atom_offsets2 = torch.stack([
        W0["embed.categorical.unit.weight"][unit] -
        W0["embed.categorical.unit.weight"][RECON_UNITS[code]] +
        W0["embed.ekind.weight"][ek] -
        W0["embed.ekind.weight"][RECON_EKINDS[code]]
        for unit, ek in combinations2
    ])
    aa2 = (aa0[None]+atom_offsets2[:, None]).reshape(-1, 192)
    cc2 = recon_context[ii].repeat(len(combinations2), 1)
    size2 = recon_lin(torch.full((len(aa2), 1), float(np.log(2))), "encode.size")
    results2 = []
    for act1, act2 in itertools.product(["gelu", "silu", "relu"], repeat=2):
        mid2 = recon_mlp(aa2, "encode.mlp1", act1) + size2
        ee2 = recon_norm(recon_mlp(mid2, "encode.mlp2", act2), "encode.norm")
        pred2 = recon_mlp(torch.cat([cc2, ee2, torch.zeros_like(ee2)], dim=1),
                          POST_PREFIX)[:, :32].reshape(len(combinations2), len(ii), 32)
        loss2 = ((pred2-target2[None])**2).mean(dim=(1, 2)).sqrt().numpy()
        k2 = int(np.argmin(loss2))
        results2.append((float(loss2[k2]), act1, act2, combinations2[k2]))
    encoder_search[code] = sorted(results2)[:4]
    print(code, "non-boundary n", len(ii),
          "(RMSE, MLP1 activation, MLP2 activation, (unit,event-kind)):",
          encoder_search[code])
print("\nCurrent pure-start reconstruction RMSE:")
for drug, rows0 in pure_action_rows.items():
    rows0 = np.asarray(rows0, dtype=int)
    print(drug, float(np.sqrt(np.mean(recon_errors[rows0]**2))))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 25: observation activation and event-kind audit\n"
            + str(encoder_search) +
            "\nNo edits. Python runs: 25; forward passes: 3.\n")
```
