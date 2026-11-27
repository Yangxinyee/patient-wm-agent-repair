Increasing the validated MAP correction to gain 8 raises arterial dose sensitivity while local factual loss remains within budget. Gain 10 exceeds the tuning limit. I’ll back up the current edit and install gain 8 for a paired GPU test; factual loss above 0.005 or no rollout improvement would make me restore the backup.

```python
import json

_r225_columns = pd.read_csv("/data/tuning.csv", nrows=1).columns.tolist()
_r225_idcol = next(
    (_c for _c in ["stay_id", "stay", "icu_stay_id"] if _c in _r225_columns),
    None)
if _r225_idcol is None:
    raise RuntimeError("Need to identify stay column: " + repr(_r225_columns))

_r225_ids = pd.read_csv(
    "/data/tuning.csv", usecols=[_r225_idcol]
)[_r225_idcol].drop_duplicates().tolist()
_r225_ids = [
    _x.item() if isinstance(_x, np.generic) else _x
    for _x in _r225_ids
]
_r225_test_ids = _r225_ids[:100]

_r225_backup = {
    _n: np.asarray(load(_n), dtype=np.float32).copy()
    for _n in head177_names
}
np.savez("/work/validated_forward21.npz", **_r225_backup)

for _n in head177_names:
    save_edit(_n, _r224_candidates[8.0]["weights"][_n])
np.savez("/work/map_gain8_candidate225.npz", **{
    _n: np.asarray(load(_n), dtype=np.float32).copy()
    for _n in head177_names
})

print("Installed gain 8; edited tensors:", edited())
print("Selected local scores:")
print(_r224_table[_r224_table.gain == 8.0].to_string(index=False))
print("NEXT FACTUAL REQUEST:")
print(json.dumps({"type": "factual", "model": "both", "split": "tuning",
                  "stays": _r225_test_ids}))
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 225: gain-8 MAP candidate installed\n"
             "- Backed up validated forward-21 edit to validated_forward21.npz.\n"
             "- Installed stronger validated MAP direction; insulin construction "
             "unchanged. Only four value-head tensors edited.\n"
             "- Local CV/tuning value dNLL +0.002336/+0.004029.\n"
             "- Pending paired GPU factual and dose-rollout checks; restore if "
             "factual tolerance fails or rollout benefit is absent.\n")
```