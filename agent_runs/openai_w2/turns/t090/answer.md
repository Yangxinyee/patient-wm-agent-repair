Changing measurement units alone did not resolve the error, so that explanation is insufficient. Since the same observation encoder reproduces END bundles accurately, I’ll test the remaining categorical tags. Consistently near-zero posterior error across sensors would identify the missing convention.

```python
import itertools

print("=== INPUT METADATA ===")
print(Path("/data/README.md").read_text()[:3000])
print("Categorical shapes:", {
    k: SHAPES[k] for k in NAMES
    if k in ("embed.kind.weight", "embed.ekind.weight", "embed.kq.weight")
})
print("Vocab keys:", list(VOCAB.keys()))

categorical_search_results = []
for event, cases in measurement_cases.items():
    pos = np.array([j for j, row in cases], dtype=int)
    base_atoms = torch.stack([embed_record_atom(row) for j, row in cases])
    original_unit = saved_unit_ids[event]
    target = sx_repair[
        torch.tensor(posterior_rows_repair[pos], dtype=torch.long), 960:992
    ]
    ncase = len(cases)
    choices = list(itertools.product(
        range(SHAPES["embed.categorical.unit.weight"][0]),
        range(SHAPES["embed.ekind.weight"][0]),
        range(SHAPES["embed.kq.weight"][0])
    ))
    # The current helper uses WORLD=0, measurement event-kind=1, kq=0.
    for begin in range(0, len(choices), 24):
        batch_choices = choices[begin:begin+24]
        variants = []
        for unit_id, event_kind, kq in batch_choices:
            offset = (
                Wrepair["embed.categorical.unit.weight"][unit_id]
                - Wrepair["embed.categorical.unit.weight"][original_unit]
                + Wrepair["embed.ekind.weight"][event_kind]
                - Wrepair["embed.ekind.weight"][1]
                + Wrepair["embed.kq.weight"][kq]
                - Wrepair["embed.kq.weight"][0]
            )
            variants.append(base_atoms + offset)
        atom_batch = torch.cat(variants, dim=0)
        e_batch = batched_set_repair(atom_batch[:, None, :], "encode")
        inp = torch.cat((
            posterior_ctx_repair[pos].repeat(len(batch_choices), 1),
            e_batch,
            posterior_u_repair[pos].repeat(len(batch_choices), 1)
        ), dim=1)
        hidden = torch.nn.functional.gelu(
            inp @ Wrepair["z_post.0.weight"].T +
            Wrepair["z_post.0.bias"]
        )
        z_pred = (
            hidden @ Wrepair["z_post.2.weight"].T +
            Wrepair["z_post.2.bias"]
        )[:, :32].reshape(len(batch_choices), ncase, 32)
        errors = torch.sqrt(
            ((z_pred - target[None, :, :]) ** 2).mean(dim=(1, 2))
        ).numpy()
        categorical_search_results.extend(
            (event, *choice, float(error))
            for choice, error in zip(batch_choices, errors)
        )

categorical_search_table = pd.DataFrame(
    categorical_search_results,
    columns=["event", "unit", "event_kind", "kq", "z_rmse"]
)
print("\n=== BEST CATEGORICAL CONVENTIONS ===")
print(categorical_search_table.sort_values("z_rmse")
      .groupby("event", sort=False).head(3)
      .sort_values(["event", "z_rmse"]).to_string(index=False))

print("\n=== EXACT MATCH COUNT ===")
print(categorical_search_table.groupby("event").z_rmse.min())
print("Current edits:", edited())
```
