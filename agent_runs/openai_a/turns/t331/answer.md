The sharper gate predicts insulin shifts of −9.067 and +3.952, versus −9.507 and +2.836 for the validated gate. That is a trade-off, not a clear improvement. I’ll recover the record-based target and the tuning IDs before broader testing. The sharper gate remains provisional until both its dose responses and factual fit are supported.

```python
import re

print("=== RECORD-BASED IDENTIFICATION CONCLUSIONS ===")
sections282 = re.split(r"(?m)(?=^## )", notes_text)
wanted282 = [
    "Sensor-contrast local projections",
    "Dynamic identification;",
    "Recovery and threshold identification",
    "Sustained insulin target and NE flexible adjustment",
]
for key in wanted282:
    for section in sections282:
        if section.startswith("## " + key):
            print(section[:1750])

print("=== TUNING ID BOUNDARIES ===")
tuning_ids282 = sorted(pd.read_csv("/data/tuning.csv", usecols=["stay_id"])
                       ["stay_id"].unique().tolist())
print("n:", len(tuning_ids282))
for start in [0, 100, 200, 300]:
    chunk = tuning_ids282[start:start+100]
    print(start, "first:", chunk[:3], "last:", chunk[-3:],
          "consecutive:", all(b == a + 1 for a, b in zip(chunk, chunk[1:])))

print("=== GRAFT OUTPUT COLUMNS ===")
for label, cp in [("gated8", validated_gated8_checkpoint),
                  ("stronger6", stronger6_checkpoint)]:
    w = np.asarray(cp["attr_heads.value_bin.2.weight"])
    print(label, "columns 41,109:", np.round(w[:, [41, 109]], 5).tolist())
    print(label, "bias minus original:",
          np.round(np.asarray(cp["attr_heads.value_bin.2.bias"]) -
                   np.asarray(load_original("attr_heads.value_bin.2.bias")), 6).tolist())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Sharp240 recovery; Python 282, forwards 36/50\n"
        "- Current tensors match sharp240_checkpoint exactly; trial remains provisional.\n"
        "- First100 exact rollout: insulin x2 -9.067, half +3.952; "
        "NE x2 -.516, half -.617. Gated8 comparison: -9.507, +2.836.\n"
        "- First100 factual delta +.00476; broader factual safety is not established.\n"
        "- Tool summary averages reached stay-hours; stay-weighted alternative was "
        "computed only as a diagnostic and is not substituted for the tool readout.\n"
        "- No additional weights changed during recovery.\n"
    )
```
