That output repeats the earlier failure, so there is still no new test result. I’ll remove the tuple ambiguity by computing context directly from the exported states. The same near-exact posterior match remains the confirmation criterion.

```python
print("RUN TAG: direct-context empty-action test")

r52_results = []
r52_best = {}

for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    frame = r45_measure_fixture.loc[
        r45_measure_fixture["event"].eq(event)
    ].head(100).copy()
    idx = frame["state_index"].to_numpy(dtype=int)
    assert np.all(r45_bank["stay_id"][idx] == r45_bank["stay_id"][idx - 1])

    previous = torch.as_tensor(
        r45_bank["x"][idx - 1], dtype=torch.float32)
    dt = (
        np.rint(r45_bank["hour"][idx] * 60)
        - np.rint(r45_bank["hour"][idx - 1] * 60)
    ) * 60
    dt_t = torch.as_tensor(dt, dtype=torch.float32)
    h = previous[:, 192:960].reshape(-1, 4, 192)
    h = h * torch.exp(
        -dt_t[:, None, None] * torch.exp(rW["core.flow.a"])[None, None, :])
    ctx = (
        previous[:, :192]
        + rp_norm(h.mean(dim=1), "core.out_norm")
        + rp_mlp(torch.log1p(dt_t)[:, None] / 10, "gap")
    )
    target = torch.as_tensor(
        r45_bank["x"][idx, 960:992], dtype=torch.float32)
    assert isinstance(ctx, torch.Tensor)

    event_best = None
    for unit in range(SHAPES["embed.categorical.unit.weight"][0]):
        for side in ["left", "right"]:
            atoms = atom_embedding_repair(
                event, values=frame["value"].to_numpy(),
                unit=unit, rate_unit=0, side=side)
            pooled = rp_mlp(atoms, "encode.mlp1")
            pooled = pooled + rp_linear(
                torch.full((len(frame), 1), np.log(2), dtype=torch.float32),
                "encode.size")
            e = rp_norm(rp_mlp(pooled, "encode.mlp2"), "encode.norm")

            for empty_name, empty_u in r50_empty_u.items():
                u = empty_u.expand(len(frame), -1)
                assert all(isinstance(v, torch.Tensor) for v in (ctx, e, u))
                z = rp_mlp(torch.cat((ctx, e, u), dim=1), "z_post")[:, :32]
                error = float(torch.mean((z - target) ** 2).sqrt())
                recipe = (error, unit, side, empty_name)
                r52_results.append((event, len(frame), *recipe))
                if event_best is None or error < event_best[0]:
                    event_best = recipe
    r52_best[event] = event_best

r52_table = pd.DataFrame(
    r52_results,
    columns=["event", "n", "RMSE", "unit", "side", "empty_action"])
print("=== BEST RECIPES ===")
for event, recipe in r52_best.items():
    print(event, recipe)
print("\n=== BEST BY EMPTY-ACTION CONVENTION ===")
print(r52_table.sort_values("RMSE").groupby(
    ["event", "empty_action"], sort=False).head(1).to_string(index=False))

print("\n=== NUISANCE MODEL TYPES ===")
for drug, models in dml_models.items():
    first = models[0]
    objects = first if isinstance(first, (tuple, list)) else [first]
    print(drug, [type(m).__name__ for m in objects])
print("current edits:", edited())
```
