The test confirms that empty action bundles cannot be represented as zero in this implementation. Some observation-vector comparisons disagree sharply, but those examples include infusion endings, so the test may have used the wrong END attributes. I’ll separate clean readings from endings and locate the atom-count mismatches before using these features for an edit.

```python
print("=== RETAINED EXACT-ENCODING OBJECTS ===")
for k71, v71 in list(globals().items()):
    if k71.endswith(("57", "58", "59", "60", "65")):
        if isinstance(v71, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(k71, type(v71).__name__, tuple(v71.shape))
        elif isinstance(v71, (dict, list, tuple)):
            print(k71, type(v71).__name__, len(v71))
        elif isinstance(v71, str) and len(v71) > 80:
            print(k71, "source string", len(v71))

print("\n=== ATOM-COUNT MISMATCHES ===")
count71 = np.array([len(rows71) for rows71 in bundle_rows65])
mismatch71 = np.flatnonzero(count71 != state_file69["n_atoms"])
print("count:", len(mismatch71))
for i71 in mismatch71[:4]:
    sid71 = int(state_file69["stay_id"][i71])
    hr71 = float(state_file69["hour"][i71])
    print("index/stay/hour:", i71, sid71, hr71,
          "expected/assembled:", int(state_file69["n_atoms"][i71]), count71[i71])
    print("assembled:", bundle_rows65[i71])
    print("nearby records:",
          tuning[(tuning.stay_id == sid71) &
                 ((tuning.hour - hr71).abs() < 0.01)].to_dict("records"))

print("\n=== CLEAN READING ENCODER CHECK ===")
checks71 = []
for i71 in valid70:
    obs71 = [(k71, r71) for k71, r71 in bundle_rows65[i71]
             if not(event_info57[r71["event"]]["is_action"] and k71 == 0)]
    if len(obs71) == 1 and obs71[0][0] == 0 and \
       obs71[0][1]["event"] in ("glucose_finger", "glucose_lab",
                               "map_arterial", "map_cuff"):
        atom71 = complete_atom57(obs71[0][1])
        e71 = single_set57(atom71[None], "encode")[0]
        z71 = post_mean59(latent_context([i71]), e71[None],
                         u_all65[i71:i71+1])
        checks71.append({
            "index": i71, "event": obs71[0][1]["event"],
            "encoder_maxdiff": float((e71 - e_all65[i71]).abs().max()),
            "posterior_rms": float(
                (z71[0] - torch.from_numpy(state_file69["x"][i71, 960:]))
                .square().mean().sqrt())
        })
        if len(checks71) == 6:
            break
print(pd.DataFrame(checks71).to_string(index=False))

print("\n=== CATEGORY MAPPINGS ===")
print(ekind_ids57)
print(unit_ids57)
print(rate_unit_ids57)
print("cached vs original action norm:",
      float((repair_weights["encode_action.norm.weight"] -
             torch.as_tensor(load_original("encode_action.norm.weight")))
            .abs().max()))
```
