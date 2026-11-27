The helpers now have the correct units and the saved contexts needed for a complete check. I’ll rebuild every noninitial bundle, including infusion END events and encoded empty sets. Posterior errors near the existing numerical tolerance across bundle types would validate this reconstruction; larger structured errors would refute it.

```python
repair_obs_sets101 = []
repair_act_sets101 = []
repair_start_rows101 = []

for _m101 in meta85.itertuples(index=False):
    _key101 = (int(_m101.stay_id), round(float(_m101.hour), 4))
    _starts101 = start_lookup_repair.get(_key101, [])
    _ends101 = end_lookup_repair.get(_key101, [])
    _obs101, _act101 = [], []
    for _r101 in _starts101:
        _a101 = embed_record_atom_v2(_r101)
        if _r101["event"] in ("insulin", "norepinephrine"):
            _act101.append(_a101)
        else:
            _obs101.append(_a101)
    _obs101.extend(embed_end_repair(_r101) for _r101 in _ends101)
    repair_obs_sets101.append(_obs101)
    repair_act_sets101.append(_act101)
    repair_start_rows101.append(_starts101)

repair_e101 = encode_variable_sets_repair(
    repair_obs_sets101, "encode", zero_empty=False)
repair_u101 = encode_variable_sets_repair(
    repair_act_sets101, "encode_action", zero_empty=False)
repair_ctx101, repair_hflow101 = previous_context_repair(posterior_rows_repair)
repair_input101 = torch.cat([repair_ctx101, repair_e101, repair_u101], dim=1)
repair_phi101 = act_fn(rlinear(repair_input101, "z_post.0"), "gelu")
repair_z101 = rlinear(repair_phi101, "z_post.2")[:, :32]
repair_truth101 = sx_repair[posterior_rows_repair, 960:992]
repair_err101 = ((repair_z101 - repair_truth101) ** 2).mean(1).sqrt().numpy()

repair_audit101 = meta85.copy()
repair_audit101["has_action"] = (
    repair_audit101.has_insulin | repair_audit101.has_norepinephrine)
repair_audit101["rmse"] = repair_err101
print("=== COMPLETE POSTERIOR RECONSTRUCTION ===")
print(repair_audit101.groupby("has_action")["rmse"].agg(["size", "mean", "max"]))
print("\nWorst bundles:")
print(repair_audit101.nlargest(8, "rmse").to_string(index=False))

print("\n=== EXPOSURE SCALE CHECK ===")
print(P[["insulin", "norepinephrine"]].describe().loc[["mean", "50%", "max"]])
print("Stored dose feature means:", dose_actual_repair.mean(axis=0))
print(drug_rows.groupby("event")[["dose", "rate"]].mean())

with open("/work/notes.md", "a") as _notes101:
    _notes101.write(
        "\n## Runs 97–101: fourth context recovery and unit correction\n"
        "Recovered architecture/helpers; no weights edited. Correct measurement unit IDs: "
        "glucose_finger=0, glucose_lab=5, map_arterial=map_cuff=2. "
        "Rebuilt complete noninitial bundles with END atoms in observations and encoded "
        "empty sets in both encoders. Reconstruction summary:\n"
        + repair_audit101.groupby("has_action")["rmse"]
          .agg(["size", "mean", "max"]).to_string() + "\n"
    )
print("\nCurrent edits:", edited())
```