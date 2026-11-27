Bundle counts now match, but the encoded states still differ substantially, so the reconstruction is not validated. The earlier END-atom search may explain the mismatch. I’ll compare its best conventions with the stored features. Close prelude and posterior agreement—not counts alone—is the required confirmation.

```python
print("EARLIER END-CONVENTION SEARCH")
print(end_route_table_repair.sort_values(
    end_route_table_repair.columns[-1]).head(8).to_string(index=False))

def bundle_inputs190(records, meta, end_kind=0):
    starts, ends = {}, {}
    for row in records.to_dict("records"):
        key = (int(row["stay_id"]), int(round(float(row["hour"]) * 60)))
        starts.setdefault(key, []).append(row)
        if pd.notna(row["end_hour"]):
            end_key = (int(row["stay_id"]),
                       int(round(float(row["end_hour"]) * 60)))
            ends.setdefault(end_key, []).append(row)
    obs, acts, action_records = [], [], []
    for row in meta.itertuples(index=False):
        key = (int(row.stay_id), int(round(float(row.hour) * 60)))
        world = starts.get(key, [])
        actions = [r for r in world
                   if r["event"] in ("insulin", "norepinephrine")]
        observations = [
            embed_record_atom_v2(r, 0, "same") for r in world
            if r["event"] not in ("insulin", "norepinephrine")]
        observations += [
            embed_record_atom_v2(r, end_kind, "same")
            for r in ends.get(key, [])]
        obs.append(observations)
        acts.append([embed_record_atom_v2(r, 0, "same") for r in actions])
        action_records.append(actions)
    return obs, acts, action_records

obs_old190, acts_old190, rows_old190 = bundle_inputs190(
    reference_records_repair, sm_repair, end_kind=0)
e_check190 = encode_variable_sets_repair(
    [obs_old190[i] for i in checks188], "encode")
u_check190 = encode_variable_sets_repair(
    [acts_old190[i] for i in checks188], "encode_action", zero_empty=True)

print("\nOLD PRELUDE MATCH TESTS")
old_actual_z190 = sx_repair[checks188, 960:992]
for label, ee, zz in [
    ("cached e, cached z", posterior_e_repair[positions188],
     posterior_z_repair[positions188]),
    ("END-kind0 e, cached z", e_check190,
     posterior_z_repair[positions188]),
    ("END-kind0 e, actual z", e_check190, old_actual_z190)]:
    eret = prelude_features111(
        hf_check189, ee, u_check190, zz)[1].numpy()
    print(label, "E RMS vs Efull134:",
          float(np.sqrt(np.mean((eret-Efull134[positions188])**2))))

obs190, acts190, action_rows190 = bundle_inputs190(
    records188, meta188, end_kind=0)
e190 = encode_variable_sets_repair([obs190[i] for i in rows188], "encode")
u190 = encode_variable_sets_repair(
    [acts190[i] for i in rows188], "encode_action", zero_empty=True)
z_fit190 = rmlp(torch.cat([ctx188, e190, u190], dim=1), "z_post")[:, :32]
err190 = torch.sqrt(torch.mean((z_fit190-z188)**2, dim=1)).numpy()
print("\nNEW END-kind0 posterior RMS / max:",
      float(np.sqrt(np.mean(err190**2))),
      float(torch.max(torch.abs(z_fit190-z188))))
print("Posterior-error quantiles:",
      np.quantile(err190, [0, .5, .9, .95, .99, 1]))
print(meta188.iloc[rows188[np.argsort(err190)[-6:]]].to_string(index=False))

print("\nFEATURE OBJECTS FROM THE CORRECTED CONSTRUCTION")
for name, obj in list(globals().items()):
    if (name.endswith(("111", "134")) and
        any(k in name.lower() for k in ["e", "z", "post", "prelude"])):
        shape = getattr(obj, "shape", None)
        if shape is not None:
            print(name, tuple(shape))
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj.keys())[:6])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run190: END-convention audit\n"
            "Counts alone did not validate reconstruction. Testing earlier "
            "best END routing/embedding convention against corrected Efull134 "
            "and original posterior states. No weight changes.\n")
```