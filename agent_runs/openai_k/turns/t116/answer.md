The reconstructed head uses `gap(log(dt)/10)` for both the prior and the readout, while the architecture specifies different scaling for the readout. I’ll check the saved test inputs and exact reading alignment before testing this. A sharp reduction in matched-reading error would support a head-construction bug.

```python
print("=== MATCHED-READING TABLE ===")
print("audit97 columns:", list(audit97.columns))
print(audit97.head(4).to_string(index=False))
print("direct98 columns:", list(direct98.columns))
print(direct98.head(3).to_string(index=False))

print("\n=== HEAD TEST OBJECTS ===")
for n109 in sorted(list(globals())):
    if n109.endswith("104") or n109.endswith("105"):
        o109 = globals()[n109]
        if hasattr(o109, "shape"):
            print(n109, type(o109).__name__, tuple(o109.shape))
        elif isinstance(o109, (list, dict)):
            print(n109, type(o109).__name__, "length", len(o109))
print("r49_noise:", tuple(r49_noise.shape))
print("r49_sensors:", r49_sensors)
print("ix104 first:", ix104[:8], "js104 first:", js104[:8],
      "m104 first:", m104[:8])
print("hours at ix104:", r49_hours[ix104[:8]])
print("base104/base105 maximum difference:",
      float((base104 - base105).abs().max()))

print("\n=== PREVIOUS CONVENTION TESTS ===")
if "results104" in globals():
    print(sorted(results104, key=lambda x: x[-1])[:8])
print("run105 best:", results105[:4])

print("\n=== IDENTIFICATION AND DYNAMICS NOTES ===")
print(text108[2800:6600])

print("\n=== FORWARD ARTIFACTS ===")
for p109 in sorted(Path("/work").glob("fwd_*")):
    print(p109.name, [q.name for q in p109.iterdir()])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 109: head-gap and alignment recovery\n"
            "- Local readout shares the prior gap scaling; architecture specifies a distinct readout scaling.\n"
            "- Checking exact matched-reading inputs before testing this candidate reconstruction fix.\n"
            "- No edits. Python runs: 109; forward passes: 4.\n")
```