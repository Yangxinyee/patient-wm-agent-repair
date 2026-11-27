The record analyses support insulin near −1.8 to −2.0 per rate unit and norepinephrine near +3.6, with additional assumptions needed for persistence. I’ll verify the decoder-to-weight mapping before making a half-dose trial. Exact reconstruction would confirm that only the intended component changes.

```python
print("TARGET-EVIDENCE SECTIONS")
for prefix230 in (
    "Run 12:", "Run 38:", "Run 39:", "Runs 157",
    "Runs181", "Runs 181"
):
    found230 = [
        s for s in notes228.split("\n## ")
        if s.startswith(prefix230)
    ]
    for s in found230[:1]:
        print("\n" + s[:1300])

print("\nFACTOR AND CURRENT-WEIGHT CHECKS")
original230 = load_original("core.b_in.weight").astype(np.float64)
current230 = load("core.b_in.weight").astype(np.float64)
print("current vs validated", np.max(np.abs(current230 - validated_B225)))
print("insulin factor error",
      np.max(np.abs(delta_ins218 - np.outer(V218[:, 0], W218[:, 0]))))
print("NE factor error",
      np.max(np.abs(delta_ne218 - np.outer(V218[:, 1], W218[:, 1]))))
print("mapped base vs insulin factor",
      np.max(np.abs(decoder_to_weight226(coef_base226) - W218[:, 0])))
print("validated gain reconstruction",
      np.max(np.abs(current230 -
                    (original230 + 1.20*delta_ins218 + 1.05*delta_ne218))))

print("\nBLEND DIAGNOSTICS")
print(blend_table226.to_string(index=False))
print("decoder function argument names",
      decoder_to_weight226.__code__.co_varnames)
print("decoder function global names",
      decoder_to_weight226.__code__.co_names)

print("\nTUNING IDENTIFIERS")
ids230 = sorted(int(s) for s in all_tuning_ids183)
print("count", len(ids230), "first", ids230[:4], "last", ids230[-4:])
print("forward38 stay range",
      int(roll225["stay_id"].min()) if "stay_id" in roll225 else roll225.columns.tolist())
```
