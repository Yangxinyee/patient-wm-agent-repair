Infusion endings now reconstruct accurately: they enter the observation encoder, and empty action bundles have a nonzero encoding. Ordinary readings still differ. I’ll isolate single-reading bundles to identify their conventions without mixed-bundle ambiguity. Near-exact agreement for each sensor would validate the remaining input reconstruction.

```python
print("=== ORDINARY-BUNDLE ERRORS BY COMPOSITION ===")
composition_errors_audit = {}
for i in ordinary_world_audit:
    composition = tuple(sorted(row.event for row, kind
                               in bundle_records_audit[state_keys_audit[i]]))
    composition_errors_audit.setdefault(composition, []).append(posterior_errors_audit[i])
for composition, errors in sorted(composition_errors_audit.items(),
                                  key=lambda item: -len(item[1]))[:12]:
    print(composition, "n =", len(errors),
          "RMS =", float(np.sqrt(np.mean(np.square(errors)))))

sensor_candidates_audit = {}
for event in measure_codes:
    single_indices = np.array([
        i for i in ordinary_world_audit
        if len(bundle_records_audit[state_keys_audit[i]]) == 1
        and bundle_records_audit[state_keys_audit[i]][0][0].event == event
    ], dtype=int)
    print("\n", event, "single-atom bundles:", len(single_indices))
    if len(single_indices) == 0:
        continue
    if len(single_indices) > 80:
        single_indices = rng_audit.choice(single_indices, 80, replace=False)
    options = []
    for unit, kq, side, value_mode in itertools.product(
            range(6), (-1, 0, 1, 2, 3), ("left", "right"),
            ("normal", "no_flag", "no_number", "absent")):
        atoms = []
        for i in single_indices:
            row = bundle_records_audit[state_keys_audit[i]][0][0]
            spec = num_spec0[(event, "value")]
            z = np.clip((row.value - spec["median"]) / spec["iqr"], -10, 10)
            nums = torch.zeros(6)
            if value_mode in ("normal", "no_flag"):
                nums[0] = z
            if value_mode in ("normal", "no_number"):
                nums[1] = 1
            atom = (W0["embed.code.weight"][code_id0[event]]
                    + W0["embed.kind.weight"][0]
                    + W0["embed.ekind.weight"][1]
                    + W0["embed.categorical.unit.weight"][unit]
                    + lin0(nums, "embed.numeric"))
            if kq >= 0:
                atom += W0["embed.kq.weight"][kq]
            if value_mode != "absent":
                bb = 1 + np.searchsorted(spec["fine_bin_edges"], row.value, side=side)
                atom += W0["embed.bins.value.weight"][bb]
            atoms.append(atom)
        ee = encode_single_audit(torch.stack(atoms), "encode", "gelu")
        uu = empty_action_audit.expand(len(ee), -1)
        pp = mlp0(torch.cat((contexts_audit[single_indices], ee, uu), 1),
                  "z_post", "gelu")[:, :32]
        error = float((pp - torch.tensor(sx[single_indices, 960:]))
                      .square().mean().sqrt())
        options.append((error, unit, kq, side, value_mode))
    sensor_candidates_audit[event] = sorted(options)[:5]
    print("Best candidates:", sensor_candidates_audit[event])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 19: sensor-specific reconstruction audit\n"
            "- END atoms enter the world encoder with absent attributes; empty action sets are encoded rather than zero.\n"
            "- Action-bundle posterior RMS now below 2e-5, but ordinary-reading inputs still need resolution.\n"
            f"- Single-sensor convention candidates: {sensor_candidates_audit}\n"
            "- Python runs: 19; forward passes: 4; no weights edited.\n")
```