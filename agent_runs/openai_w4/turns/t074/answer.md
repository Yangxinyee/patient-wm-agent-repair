The recipes already reproduce ordinary measurements and many infusion bundles accurately. I’ll check whether the outliers involve simultaneous infusion endings, unusually large doses, or the stay-ending event. If their counts and numeric values explain the errors, we can fix the reconstruction locally; if not, I’ll revisit the encoder calculation.

```python
def arr64(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

count_difference64 = (
    arr64(counts_e58) + arr64(end_count60) + arr64(counts_u58)
    - arr64(repair_bank["n_atoms"])
)
bad_count64 = np.flatnonzero(count_difference64)
print("count mismatch summary:", pd.Series(count_difference64[bad_count64]).value_counts().to_dict())
print("first count mismatches:")
for j in bad_count64[:5]:
    print(int(j), int(repair_bank["stay_id"][j]), float(repair_bank["hour"][j]),
          int(repair_bank["n_atoms"][j]), int(counts_e58[j]), int(end_count60[j]), int(counts_u58[j]))
print("\nworst posterior cases:")
for p64 in np.argsort(err60)[-8:][::-1]:
    j64 = int(valid58[p64])
    sid64 = int(repair_bank["stay_id"][j64])
    hour64 = float(repair_bank["hour"][j64])
    print("\nSTATE", j64, sid64, hour64, "RMSE", float(err60[p64]),
          "counts", int(repair_bank["n_atoms"][j64]), int(counts_e58[j64]),
          int(end_count60[j64]), int(counts_u58[j64]))
    print(repair_records.loc[
        (repair_records.stay_id == sid64) &
        (repair_records.hour.between(hour64 - 1.01, hour64 + .001)),
        ["hour", "event", "value", "rate", "dose", "end_hour"]
    ].to_string(index=False))
    print("END:", end_records60.loc[
        end_records60.end_bundle == j64, ["event", "rate", "dose", "end_hour"]
    ].to_dict("records"))
print("\nfunction constants:")
for name64 in ["atom_clean60", "atom_embedding_repair", "pooled_encoder58", "pre_context_repair"]:
    fn64 = globals()[name64]
    print(name64, "names=", fn64.__code__.co_names, "constants=",
          [c for c in fn64.__code__.co_consts if isinstance(c, (str, int, float, tuple))])
print("bank shapes:", {k: np.shape(v) for k,v in repair_bank.items()})
```
