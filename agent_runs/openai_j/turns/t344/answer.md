This repeats run 290’s truncated output, so there is no new validation result to interpret. I’ll request a shorter audit to identify which stay groups have already been tested and recover the edit’s rationale. That will prevent duplicate forward passes and show whether the remaining checks can meaningfully challenge the current edit.

```python
import re

print("=== VALIDATION GROUPS ===")
for final_dir291 in ["fwd_40", "fwd_42", "fwd_43",
                     "fwd_44", "fwd_45", "fwd_46"]:
    final_p291 = Path("/work") / final_dir291
    if not final_p291.exists():
        continue
    for final_file291 in sorted(final_p291.glob("*_edited.csv")):
        final_d291 = pd.read_csv(final_file291)
        final_ids291 = sorted(final_d291["stay_id"].unique())
        print(final_dir291, final_file291.name, len(final_ids291),
              final_ids291[0], final_ids291[-1])
        if "dose_edit" in final_file291.name:
            print("columns:", list(final_d291.columns))
    final_sp291 = final_p291 / "summary.json"
    if final_sp291.exists():
        final_s291 = json.loads(final_sp291.read_text())
        print("metadata:", {
            k: v for k, v in final_s291.items()
            if not isinstance(v, (dict, list))
        })
        def final_walk291(obj, path=""):
            if isinstance(obj, dict):
                if "mean_shift" in obj:
                    print(path, obj["mean_shift"],
                          "hours", obj.get("n_stay_hours"))
                else:
                    for key, val in obj.items():
                        final_walk291(val, path + "/" + str(key))
        final_walk291(final_s291)

print("\n=== NE SENSOR-IV ESTIMATES ===")
print(pd.read_csv("/work/sensor_iv.csv").tail(8).to_string(index=False))

final_text291 = Path("/work/notes.md").read_text()
final_sections291 = re.split(r"(?m)(?=^## )", final_text291)
final_method291 = [
    s for s in final_sections291
    if any(t in s.split("\n", 1)[0].lower()
           for t in ("kernel", "causal", "residual",
                     "intervention", "repair", "safety"))
]
print("\n=== METHOD NOTES (BOUNDED) ===")
final_selected291 = final_method291[:3] + final_method291[-3:]
for final_sec291 in dict.fromkeys(final_selected291):
    print(final_sec291[:500])
print("\n=== CANDIDATE 285 NOTES ===")
for final_sec291 in final_sections291:
    if "285" in final_sec291.split("\n", 1)[0]:
        print(final_sec291[:900])

print("\n=== RECOVERY HELPERS ===")
print(sorted(
    k for k, v in final_globals290.items()
    if callable(v) and not k.startswith("final_")
    and any(t in k.lower() for t in
            ("candidate", "calib", "restore", "kernel", "safety", "head"))
)[:35])
print("Current edits:", edited())

with open("/work/notes.md", "a") as final_notes291:
    final_notes291.write(
        "\n## Compact final evidence recovery\n"
        "- Latest received output repeated run 290; no new forward result.\n"
        "- Recovering exact validation groups and estimation rationale "
        "with bounded output. No weights changed.\n"
        "- Last externally confirmed counts before this run: "
        "290 Python runs, 44 forward passes.\n"
    )
```