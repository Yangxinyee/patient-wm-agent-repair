The drug checks validate the shared numerical embedding, so the measurement mismatch may instead involve how an empty action bundle is represented. I’ll test that directly before rebuilding more of the model. A posterior match near numerical precision would confirm the missing convention; otherwise I’ll check time-flow alignment.

```python
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)

r50_empty_u = {
    "zero": torch.zeros(1, 192),
    "mlp2_zero": rp_norm(
        rp_mlp(torch.zeros(1, 192), "encode_action.mlp2"),
        "encode_action.norm"),
    "size_bias": rp_norm(
        rp_mlp(rp_linear(torch.zeros(1, 1), "encode_action.size"),
               "encode_action.mlp2"),
        "encode_action.norm"),
}

r50_results = []
r50_best = {}
for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    frame = r45_measure_fixture.loc[
        r45_measure_fixture["event"].eq(event)
    ].head(100).copy()
    idx = frame["state_index"].to_numpy(dtype=int)
    ctx = pre_context_repair(r45_bank, idx)
    target = torch.as_tensor(r45_bank["x"][idx, 960:992], dtype=torch.float32)
    event_best = None

    for unit in range(SHAPES["embed.categorical.unit.weight"][0]):
        for side in ["left", "right"]:
            atoms = atom_embedding_repair(
                event, values=frame["value"].to_numpy(),
                unit=unit, rate_unit=0, side=side)
            pooled = rp_mlp(atoms, "encode.mlp1")
            pooled += rp_linear(
                torch.full((len(frame), 1), np.log(2), dtype=torch.float32),
                "encode.size")
            e = rp_norm(rp_mlp(pooled, "encode.mlp2"), "encode.norm")

            for empty_name, empty_u in r50_empty_u.items():
                u = empty_u.expand(len(frame), -1)
                z = rp_mlp(torch.cat([ctx, e, u], dim=1), "z_post")[:, :32]
                error = float(torch.mean((z - target) ** 2).sqrt())
                recipe = (error, unit, side, empty_name)
                r50_results.append((event, len(frame), *recipe))
                if event_best is None or error < event_best[0]:
                    event_best = recipe
    r50_best[event] = event_best

r50_table = pd.DataFrame(
    r50_results,
    columns=["event", "n", "RMSE", "unit", "side", "empty_action"])
print("=== BEST POSTERIOR RECIPES ===")
for event, recipe in r50_best.items():
    print(event, recipe)
print("\n=== BEST RECIPE BY EMPTY-ACTION CONVENTION ===")
print(r50_table.sort_values("RMSE").groupby(
    ["event", "empty_action"], sort=False).head(1).to_string(index=False))

print("\n=== NUISANCE MODEL STRUCTURE ===")
for drug, models in dml_models.items():
    print(drug, "folds:", len(models))
    for j, fold_models in enumerate(models[:1]):
        objects = fold_models if isinstance(fold_models, (tuple, list)) else [fold_models]
        for model in objects:
            print(" ", j, type(model).__name__,
                  {k: v for k, v in model.get_params().items()
                   if k in ["max_iter", "max_leaf_nodes", "min_samples_leaf",
                            "learning_rate", "l2_regularization"]}
                  if hasattr(model, "get_params") else str(model)[:180])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 50\n"
            "Tested zero and learned empty-action encodings against singleton measurement "
            "posterior states. No weights edited; result determines whether the posterior "
            "reconstruction can be used for a targeted repair.\n")
```
