The earlier record analysis found insulin effects near −2 glucose units per dose unit and norepinephrine effects near +3 MAP units, with broadly consistent results across sensors. Those are one-hour effects, not the rollout targets. I’ll recover the dynamic estimates and available helpers next; persistent effects would imply a larger rollout response than the original model predicts.

```python
rr37_notes = Path("/work/notes.md").read_text()
rr37_start = rr37_notes.find("## Run 6:")
rr37_end = rr37_notes.find("## Run 14:")
print("=== IDENTIFICATION AND EARLY EDIT WORK ===")
print(rr37_notes[rr37_start:rr37_end][:6200])
print("\n=== PERSISTENT VARIABLE NAMES ===")
rr37_names = sorted(k for k in globals()
                    if not k.startswith("_") and not k.startswith("rr37"))
print(", ".join(rr37_names)[:3300])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 37: recover dynamic identification\n"
            "- Earlier cross-fitted one-hour estimates are about -2 for insulin/glucose and +3 for norepinephrine/MAP, with sensor agreement.\n"
            "- Recovering impulse-response estimates and existing helpers; no edit yet.\n"
            "- Python runs: 37; forward passes: 4.\n")
```