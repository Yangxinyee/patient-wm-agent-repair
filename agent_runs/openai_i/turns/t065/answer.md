The helper indeed omitted measurement values, so those failed checks did not test the model correctly. I’ll rebuild complete atom vectors and compare their posteriors with the exposed original states. Errors near the drug-start reconstruction error would confirm the encoding; substantially larger errors would identify another missing field.

```python
print("=== ABSENT-ENTRY NORMS ===")
for n in NAMES:
    if n.startswith("embed.") and len(SHAPES[n]) == 2:
        print(n, round(float(torch.linalg.vector_norm(repair_weights[n][0])), 6))

event_info57 = {x["name"]: x for x in VOCAB["codes"] if "event_kind" in x}
ekind_ids57 = {v: int(k) for k, v in VOCAB["event_kinds"].items()}
unit_ids57 = {"glucose_finger": 2, "glucose_lab": 2,
              "map_arterial": 5, "map_cuff": 5,
              "insulin": 3, "norepinephrine": 4}
rate_unit_ids57 = {"insulin": 2, "norepinephrine": 3}

def complete_atom57(row, kind=0, ekind=None, unit=None, rate_unit=None,
                    numbers=True):
    event = row["event"]
    if ekind is None:
        ekind = ekind_ids57[event_info57[event]["event_kind"]]
    v = (repair_weights["embed.code.weight"][code_ids[event]]
         + repair_weights["embed.kind.weight"][kind]
         + repair_weights["embed.ekind.weight"][ekind]).clone()
    nums = np.zeros(6, dtype=np.float32)
    for attr, offset in [("value", 0), ("dose", 2), ("rate", 4)]:
        value = row.get(attr, np.nan) if numbers else np.nan
        b = 0
        if value is not None and pd.notna(value) and (event, attr) in numeric_specs:
            spec = numeric_specs[event, attr]
            nums[offset] = np.clip((float(value) - spec["median"]) / spec["iqr"], -10, 10)
            nums[offset + 1] = 1
            b = 1 + np.searchsorted(spec["fine_bin_edges"], float(value), side="left")
        v += repair_weights[f"embed.bins.{attr}.weight"][int(b)]
    v += repair_linear(torch.from_numpy(nums), "embed.numeric")
    choices = {"unit": unit_ids57.get(event, 0) if unit is None else unit,
               "rate_unit": rate_unit_ids57.get(event, 0) if rate_unit is None else rate_unit,
               "action": 0, "status": 0, "route": 0, "discharged_to": 0}
    for attr, index in choices.items():
        v += repair_weights[f"embed.categorical.{attr}.weight"][index]
    return v

def single_set57(atom_batch, prefix="encode"):
    pooled = repair_mlp_act(atom_batch, prefix + ".mlp1", F.gelu)
    size = repair_linear(torch.full((atom_batch.shape[0], 1), np.log(2),
                                    dtype=torch.float32), prefix + ".size")
    return repair_norm(repair_mlp_act(pooled + size, prefix + ".mlp2", F.gelu),
                       prefix + ".norm")

measurement_checks57 = []
for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    cases = []
    for i in range(1, len(latent_sid)):
        if latent_sid[i] != latent_sid[i-1]:
            continue
        key = (int(latent_sid[i]), float(latent_hour[i]))
        rows = repair_lookup.get(key)
        if rows is not None and len(rows) == 1 and rows.iloc[0]["event"] == event:
            cases.append((i, rows.iloc[0].to_dict()))
            if len(cases) == 24:
                break
    if not cases:
        continue
    ii = np.array([x[0] for x in cases])
    cc = latent_context(ii)
    zz = torch.from_numpy(latent_state[ii, 960:])
    results = []
    for unit in range(6):
        aa = torch.stack([complete_atom57(r, unit=unit) for _, r in cases])
        ee = single_set57(aa)
        zp = latent_post_mean(cc, ee, torch.zeros_like(ee))
        results.append((float(torch.mean((zp - zz)**2).sqrt()), unit))
    measurement_checks57.append((event, len(cases), sorted(results)[:2]))
print("\n=== CORRECTED MEASUREMENT CHECKS ===")
for x in measurement_checks57:
    print(x)

end_cases57 = [(i, ends[0]) for i, starts, ends in latent_bundles
               if not starts and len(ends) == 1][:40]
ii = np.array([x[0] for x in end_cases57])
cc = latent_context(ii)
zz = torch.from_numpy(latent_state[ii, 960:])
end_search57 = []
for ekind in range(5):
    for numbers in [False, True]:
        for unit_mode in [0, 1, "record"]:
            for rate_mode in [0, 1, "record"]:
                aa = torch.stack([
                    complete_atom57(r, kind=2, ekind=ekind,
                                    unit=None if unit_mode == "record" else unit_mode,
                                    rate_unit=None if rate_mode == "record" else rate_mode,
                                    numbers=numbers)
                    for _, r in end_cases57])
                ee = single_set57(aa)
                zp = latent_post_mean(cc, ee, torch.zeros_like(ee))
                err = float(torch.mean((zp - zz)**2).sqrt())
                end_search57.append((err, ekind, numbers, unit_mode, rate_mode))
print("\n=== END CHECKS ===")
print(sorted(end_search57, key=lambda x: x[0])[:8])
print("Current edits:", edited())
```
