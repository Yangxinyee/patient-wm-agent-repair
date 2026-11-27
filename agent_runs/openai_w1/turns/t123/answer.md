The older “corrected” atom helper differs by nearly constant, sensor-specific offsets, which suggests categorical embeddings rather than numeric values. I’ll test those offsets and the absent-rate-unit embedding against exported posteriors. A numerical match would validate a correction; smaller but persistent errors would not be enough.

```python
def rr_encode_atoms112(atoms, mask, prefix, empty_zero=False):
    mask = torch.as_tensor(mask, dtype=torch.bool)
    owner = rr_owner4[mask]
    count = torch.bincount(owner, minlength=rr_n4).float()
    hidden = rr_mlp(atoms[mask], prefix + ".mlp1")
    summed = torch.zeros(rr_n4, hidden.shape[1])
    summed.index_add_(0, owner, hidden)
    pooled = summed / count.clamp_min(1).sqrt()[:, None]
    pooled += rr_linear(torch.log1p(count)[:, None], prefix + ".size")
    result = rr_norm(rr_mlp(pooled, prefix + ".mlp2"), prefix + ".norm")
    if empty_zero:
        result[count == 0] = 0
    return result

rr_obs_mask112 = ~rr_is_drug4 | rr_is_end4
rr_act_mask112 = rr_is_drug4 & ~rr_is_end4
rr_base_u112 = rr_encode_atoms112(rr_atoms4, rr_act_mask112, "encode_action", True)
rr_candidates112 = {"current": rr_atoms4}
rr_old_offset112 = rr_atoms4.clone()
print("=== SENSOR-SPECIFIC OLD-HELPER OFFSETS ===")
for sensor in sorted(rr_sensors):
    idx = [i for i, d in enumerate(rr_desc4) if d["code"] == sensor]
    delta = r95_correct_atom(**rr_desc4[idx[0]]) - rr_atoms4[idx[0]]
    rr_old_offset112[idx] += delta
    print(sensor, "offset norm", float(delta.norm()))
rr_candidates112["old_helper_sensor_offsets"] = rr_old_offset112

rr_ru0112 = rr_original["embed.categorical.rate_unit.weight"][0]
for sign in (1, -1):
    candidate = rr_atoms4.clone()
    candidate[~rr_is_drug4] += sign * rr_ru0112
    rr_candidates112["rate_unit0_" + str(sign)] = candidate

print("\n=== POSTERIOR CANDIDATE ERRORS ===")
rr_results112 = []
for label, atoms in rr_candidates112.items():
    e = rr_encode_atoms112(atoms, rr_obs_mask112, "encode")
    z = rr_mlp(torch.cat([rr_best_ctx111, e, rr_base_u112], 1), "z_post")[:, :32]
    err = (z-rr_zexport4).square().mean(1).sqrt().numpy()
    rr_results112.append((float(err[rr_valid4].mean()), label))
    print(label,
          "reading", np.quantile(err[rr_valid4 & rr_has_reading4], [.5, .9, 1]),
          "no-reading", np.quantile(err[rr_valid4 & ~rr_has_reading4], [.5, .9, 1]))

print("\n=== FRESH ATOM CATEGORICAL LOGIC ===")
for ins in dis.get_instructions(rr_atom):
    if ins.offset >= 650:
        print(ins.offset, ins.opname, ins.argrepr)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 112\nTesting categorical-offset explanations against exported posteriors with an explicit cache-free set encoder. No weights edited. Counts: 112 Python runs, 6 forward passes.\n")
```