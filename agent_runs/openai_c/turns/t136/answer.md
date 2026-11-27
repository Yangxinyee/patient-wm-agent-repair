Query sharpening produced only small changes: insulin −2.633/+1.688 and norepinephrine +0.134/−0.012. That does not explain the large gap to the record-based targets, so I’ll remove it. Next I’ll review the earlier selective edits and recurrent tensors to avoid repeating failed tests; a useful new edit must strengthen drug response without disturbing factual readings.

```python
undo()
print("RESTORED", ops())

print("\nEARLIER SELECTIVE EDITS")
print(notes_text[10000:16900])

print("\nACTION AND RECURRENT TENSORS")
for name in NAMES:
    if name.startswith("encode_action.") or any(
        s in name for s in ["core.b_in", "core.delta_in", "core.a_in",
                           "core.c_out", "core.prelude.0.attn"]
    ):
        print(name, SHAPES[name])

print("\nDIRECTION AND HELPER VARIABLES")
for key, val in list(globals().items()):
    if key.startswith("_"):
        continue
    if any(s in key.lower() for s in ["direction", "sensor", "shear", "dose_dir", "gain4", "conditional"]):
        if isinstance(val, np.ndarray):
            print(key, val.shape)
        elif isinstance(val, dict):
            print(key, "keys", list(val)[:8])
        elif callable(val):
            import inspect
            print(key, inspect.signature(val))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python runs 96–97 / forward 36: query sharpening rejected\n"
            "- Recovered record-based probe targets: insulin -12.04/+6.02; NE +4.73/-2.37.\n"
            "- Prelude query x2: insulin -2.633/+1.688; NE +0.134/-0.012.\n"
            "- Improvement is marginal, with factual safety untested. Removed query scaling.\n"
            "- Numeric gain 4 alone restored. Reviewing distinct recurrent action pathways.\n"
            "- Forward passes used: 36; 14 remain.\n")
```
