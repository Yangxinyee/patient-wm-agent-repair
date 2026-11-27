The records support an insulin effect near −2 glucose units per rate-unit and a norepinephrine effect near +3 MAP units per rate-unit, with effects persisting beyond one hour. That remains the repair target, but the local evaluator is not yet trustworthy. I’ll inspect its head calculations and row-level errors; systematic errors by sensor or timestamp would identify where it diverges from the forward tool.

```python
import dis

print("=== MATCHED-READING ERRORS ===")
for sensor101, part101 in direct98.groupby("event"):
    dm101 = part101["local_mean"].to_numpy() - part101["predicted_mean"].to_numpy()
    dn101 = part101["local_nll"].to_numpy() - part101["nll"].to_numpy()
    print(sensor101, "n", len(part101),
          "mean error quantiles", np.round(np.quantile(dm101, [0, .1, .5, .9, 1]), 4),
          "mean abs error", np.mean(np.abs(dm101)),
          "NLL difference", np.mean(dn101))
print(direct98.head(12).to_string(index=False))
print("=== FEATURE / HEAD IMPLEMENTATIONS ===")
for fname101 in ("r49_readouts", "r59_features"):
    f101 = globals()[fname101]
    print("\nFUNCTION", fname101, inspect.signature(f101))
    print("VARIABLES", f101.__code__.co_varnames)
    print("GLOBALS", f101.__code__.co_names)
    print("CONSTANTS", repr(f101.__code__.co_consts)[:1300])
    ins101 = list(dis.get_instructions(f101))
    lines101 = [f"{i.offset}:{i.opname} {i.argrepr}" for i in ins101]
    print("\n".join(" | ".join(lines101[k:k+6])
                    for k in range(0, len(lines101), 6))[:6200])
print("=== STATE FILE ===")
sfile101 = np.load("/work/fwd_3/states_original.npz")
print([(k, sfile101[k].shape, str(sfile101[k].dtype)) for k in sfile101.files])
print("=== FEATURE OBJECTS ===")
for n101 in ("features97", "fresh_mean97", "p97", "valid97", "b98",
             "fact_indices98", "all_pre98", "eligible98"):
    o101 = globals()[n101]
    print(n101, type(o101).__name__, getattr(o101, "shape", None),
          [getattr(x, "shape", None) for x in o101] if isinstance(o101, tuple) else "")
with open("/work/notes.md", "a") as f101:
    f101.write("\n## Run 101: head reconstruction audit\n"
               "- Record-derived immediate effects remain approximately insulin -2 and NE +3, with persistence.\n"
               "- Inspecting exact matched-reading errors and head/feature implementations before trusting local edit scores.\n"
               "- Python runs: 101; forward passes: 4; no submitted edits.\n")
```
