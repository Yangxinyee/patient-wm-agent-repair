The posterior has a compact layer that combines patient history with doses, making it a possible edit location. But it also processes observations, so the unresolved reconstruction error could invalidate factual-fit estimates. I’ll test measurement-unit encoding first; reconstruction error near the drug bundles’ \(4\times10^{-5}\) would support that explanation.

```python
import inspect

print("Measurement unit assignments:", unit_ids_repair)
print("Unit table shape:", SHAPES["embed.categorical.unit.weight"])
print("Embedding signature:", inspect.signature(embed_record_atom))
print("Set encoder signature:", inspect.signature(encode_atom_set))

measurement_cases = {}
for j, state_idx in enumerate(posterior_rows_repair):
    meta = sm_repair.iloc[int(state_idx)]
    if bool(meta.has_insulin) or bool(meta.has_norepinephrine):
        continue
    if int(meta.n_atoms) != 1:
        continue
    key = (int(meta.stay_id), int(round(float(meta.hour) * 3600)))
    records = world_lookup_repair.get(key, [])
    if len(records) != 1:
        continue
    row = records[0]
    event = row["event"]
    if event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
        measurement_cases.setdefault(event, [])
        if len(measurement_cases[event]) < 30:
            measurement_cases[event].append((j, row))

print("Singleton cases:", {k: len(v) for k, v in measurement_cases.items()})
unit_search_results = []
saved_unit_ids = dict(unit_ids_repair)

try:
    for event, cases in measurement_cases.items():
        positions = np.array([j for j, row in cases], dtype=int)
        target_z = sx_repair[
            torch.tensor(posterior_rows_repair[positions], dtype=torch.long),
            960:992
        ]
        for unit_id in range(SHAPES["embed.categorical.unit.weight"][0]):
            unit_ids_repair[event] = unit_id
            atoms = [[embed_record_atom(row)] for j, row in cases]
            e_test = encode_variable_sets_repair(atoms, "encode")
            inp = torch.cat((
                posterior_ctx_repair[positions],
                e_test,
                posterior_u_repair[positions]
            ), dim=1)
            phi_test = torch.nn.functional.gelu(
                inp @ Wrepair["z_post.0.weight"].T +
                Wrepair["z_post.0.bias"]
            )
            z_test = (
                phi_test @ Wrepair["z_post.2.weight"].T +
                Wrepair["z_post.2.bias"]
            )[:, :32]
            rmse = float(torch.sqrt(torch.mean((z_test - target_z) ** 2)))
            unit_search_results.append((event, unit_id, len(cases), rmse))
finally:
    unit_ids_repair.clear()
    unit_ids_repair.update(saved_unit_ids)

unit_search_table = pd.DataFrame(
    unit_search_results, columns=["event", "unit_id", "n", "z_rmse"]
)
print("\nUnit search:")
print(unit_search_table.to_string(index=False))

print("\nSynthetic-dose target ranges:")
for key, value in posterior_cf_targets_repair.items():
    x = np.asarray(value)
    print(key, "shape", x.shape, "min", x.min(axis=0),
          "max", x.max(axis=0), "nonzero", (x != 0).sum(axis=0))

print("\nReconstruction helper settings:")
for name in ("rmlp", "rnorm", "encode_atom_set"):
    func = globals()[name]
    print(name, inspect.signature(func), "constants:", func.__code__.co_consts)
```
