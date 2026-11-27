The architecture confirms that the current edit acts through the recurrent state, leaving the prediction heads unchanged. The output was truncated before the candidate details, so I’ll retrieve those and the effect-estimation notes separately. I need a reproducible initial-safe decoder before testing stronger gains.

```python
import re

print("=== RUN-147 OBJECTS AND CANONICAL EDIT COMPONENTS ===")
keys149 = list(globals())
for name149 in keys149:
    if name149.startswith("_"):
        continue
    if name149.endswith("147") or name149 in [
        "Wdose134", "V128", "fallback_recurrent139", "repair_head_cfg",
        "fit128", "tuning_ids", "train_ids"
    ]:
        obj149 = globals()[name149]
        if callable(obj149):
            try:
                desc149 = str(inspect.signature(obj149))
            except Exception:
                desc149 = type(obj149).__name__
        elif isinstance(obj149, dict):
            desc149 = {
                k: (f"shape={tuple(v.shape)}" if hasattr(v, "shape")
                    else repr(v)[:100])
                for k, v in list(obj149.items())[:25]
            }
        elif hasattr(obj149, "shape"):
            desc149 = f"{type(obj149).__name__} shape={tuple(obj149.shape)}"
        else:
            desc149 = repr(obj149)[:350]
        print(name149, desc149)

print("\n=== EARLY NOTE SECTION INDEX ===")
sections149 = re.split(r"(?m)(?=^## )", notes148)
for section149 in sections149:
    header149 = section149.splitlines()[0] if section149.strip() else ""
    match149 = re.search(r"\bRun(?:s)?\s+(\d+)", header149)
    if match149 and int(match149.group(1)) <= 45:
        print(header149)

print("\n=== CAUSAL AND PERSISTENCE SUMMARY LINES ===")
lines149 = notes148.splitlines()
matches149 = [
    i for i, line in enumerate(lines149)
    if any(term in line.lower() for term in [
        "persistence", "accumulation", "instrument", "randomiz",
        "immediate effect", "causal estimate", "record-supported", "target:"
    ])
]
shown149 = set()
used149 = 0
for i149 in matches149:
    for j149 in range(max(0, i149 - 1), min(len(lines149), i149 + 3)):
        if j149 not in shown149 and used149 < 3000:
            print(lines149[j149])
            shown149.add(j149)
            used149 += len(lines149[j149]) + 1
print("\nCurrent edited tensors:", edited())
```
