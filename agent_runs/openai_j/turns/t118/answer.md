The records support an immediate glucose-lowering insulin effect and a MAP-raising norepinephrine effect. The insulin persistence estimate is much less stable, so its long-horizon target remains provisional. I’ll recover the earlier repair attempts and inspect the numeric-register implementation; a register-layout error could explain why the local factual check failed.

```python
import dis

print("=== EARLIER REPAIR ATTEMPTS ===")
print(notes_text[10800:16000])

print("\n=== NUMERIC REGISTER IMPLEMENTATION ===")
instructions = list(dis.get_instructions(reconstruct_reading_head))
hits = [
    i for i, inst in enumerate(instructions)
    if inst.argval in ("last_numeric.weight", "last_numeric.bias", "zlast",
                       "has_last", "age_s")
]
indices = sorted({
    j for i in hits
    for j in range(max(0, i - 5), min(len(instructions), i + 9))
})
for j in indices:
    inst = instructions[j]
    print(f"{inst.offset:5d} {inst.opname:24s} {inst.argrepr}")

print("\n=== AVAILABLE MATCHED READING TABLES ===")
shown = set()
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and id(obj) not in shown:
        if {"state_i", "code"}.issubset(obj.columns):
            shown.add(id(obj))
            print(name, obj.shape, list(obj.columns))
            print(obj.head(2).to_string(index=False))
            if len(shown) == 2:
                break
```
