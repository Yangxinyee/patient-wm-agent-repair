The original action encoder has substantial, nonzero gains, so it is not simply disabled. I’ll check timing alignment and the local reconstruction errors before trusting the earlier failed decoders. If those checks pass but held-out dose decoding still fails, the problem is more likely in dose representation than in a misplaced regression target.

```python
print("Estimated dynamics:", dyn_estimates)
print("Alternative dynamics:", dynamic_alternatives)
print("Target checks:", target_checks_A)

print("Latent reconstruction errors:")
for key in ["zerr4", "row_error"]:
    if key in globals():
        a_repair = np.asarray(globals()[key])
        print(key, np.quantile(a_repair[np.isfinite(a_repair)], [0, .25, .5, .9, 1]))
if "startmask4" in globals():
    for flag in [False, True]:
        a_repair = np.asarray(zerr4)[np.asarray(startmask4) == flag]
        print("zerr4 action-start", flag, "n", len(a_repair),
              "median/max", np.median(a_repair), np.max(a_repair))

print("Correction data timing sample:")
for key in ["corrTrainA", "corrTrainB"]:
    c_repair = globals()[key]
    print(key, list(c_repair["idx"][:8]),
          "innovations", c_repair["innovations"][:8].tolist(),
          "desired", c_repair["desired"][:8].tolist())
print("Existing ridge diagnostics:")
for k, v_repair in list(globals().items()):
    if any(s in k.lower() for s in ["ridge", "r2", "score", "decoder", "decode", "alpha"]):
        if np.isscalar(v_repair) or isinstance(v_repair, (dict, list, tuple)):
            print(k, str(v_repair)[:1000])

def compact_dis_repair(f_repair):
    groups_repair = []
    current_repair = []
    for ins_repair in dis.get_instructions(f_repair):
        if ins_repair.starts_line is not None and current_repair:
            groups_repair.append(" ; ".join(current_repair))
            current_repair = []
        if ins_repair.opname not in {"RESUME", "CACHE", "NOP"}:
            current_repair.append(f"{ins_repair.opname} {ins_repair.argrepr}")
    if current_repair:
        groups_repair.append(" ; ".join(current_repair))
    return "\n".join(groups_repair)

for function_name in ["action_vectors4", "action_atom_exact",
                      "correction_data", "wm_layer"]:
    f_repair = globals()[function_name]
    print("\n", function_name, inspect.signature(f_repair))
    print(compact_dis_repair(f_repair)[:2200])
```
