The recovered history shows a stable insulin slope near −2 per rate unit, while norepinephrine estimates change substantially with the controls used. Those slopes alone do not determine the rollout targets. The output was truncated, so I’ll retrieve the longer-horizon estimates and candidate functions before testing the edit.

```python
print("=== LONG-HORIZON IDENTIFICATION ===")
rc_start = rc_notes.find("## Run 5")
rc_end = rc_notes.find("## Run 12", rc_start)
print(rc_notes[rc_start:rc_end if rc_end > rc_start else rc_start+4800][:4800])

print("\n=== TARGET / STRONG-CANDIDATE SUMMARY LINES ===")
rc_selected = [
    s for s in rc_notes.splitlines()
    if any(k in s.lower() for k in [
        "target", "true effects", "true change", "strength-2.6",
        "forward requests 27", "forward request 28", "forward 28"
    ])
]
print("\n".join(rc_selected[-25:])[:2000])

print("\n=== FUNCTION NAMES ===")
rc_fnames = [
    n for n, o in list(globals().items())
    if inspect.isfunction(o) and not n.startswith("rc_")
]
print(" ".join(rc_fnames)[-2000:])

print("\n=== RUN 232 OBJECTS ===")
for rc_name, rc_obj in list(globals().items()):
    if rc_name.startswith("r232"):
        if hasattr(rc_obj, "shape"):
            print(rc_name, type(rc_obj).__name__, rc_obj.shape)
        elif isinstance(rc_obj, dict):
            print(rc_name, "keys:", list(rc_obj.keys())[:15])
        else:
            print(rc_name, str(rc_obj)[:180])

print("\n=== OTHER RECOVERY NAMES ===")
rc_candidates = [
    n for n in globals()
    if any(k in n.lower() for k in ["target", "cache", "tune_ids", "train_ids", "strong"])
    and not n.startswith("rc_")
]
print(" ".join(rc_candidates)[-1200:])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 234: focused recovery\n"
            "Previous output was truncated. Retrieved long-horizon identification "
            "and recent candidate objects with bounded output. No weights changed.\n")
```