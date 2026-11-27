The key data and state files are still in memory. The architecture’s register description is simplified, but the actual input has eight fields; that discrepancy could explain the CPU errors. I’ll inspect the normalization note and register-building code next. A corrected layout should sharply reduce the error without changing any weights.

```python
print("=== NORMALIZATION NOTES ===")
print(VOCAB.get("numeric_note"))
print(VOCAB.get("value_bin_note"))
print("\n=== RELEVANT SHAPES ===")
for name in NAMES:
    if name.startswith(("last_", "query_", "type_", "core.z_head",
                        "core.z_prior", "core.gap", "embed.numeric")):
        print(name, SHAPES[name])
print("\n=== HEAD FUNCTION CONSTANTS ===")
for fn in [make_head_features, cpu_head_parts, cpu_head_probs, reading_metadata]:
    print(fn.__name__, "start", fn.__code__.co_firstlineno)
    print([x for x in fn.__code__.co_consts
           if isinstance(x, (str, int, float, tuple))][:65])

print("\n=== HEAD INPUT CONSTRUCTION ===")
for fn, lo, hi in [(make_head_features, 65, 105), (cpu_head_parts, 0, 1000)]:
    print("FUNCTION", fn.__name__)
    rows = []
    active_line, parts = None, []
    for ins in dis.get_instructions(fn):
        if ins.starts_line is not None and ins.starts_line != active_line:
            if active_line is not None and lo <= active_line <= hi:
                rows.append((active_line, " | ".join(parts)))
            active_line, parts = ins.starts_line, []
        if ins.opname not in {"RESUME", "CACHE"}:
            parts.append(ins.opname + ":" + ins.argrepr)
    if active_line is not None and lo <= active_line <= hi:
        rows.append((active_line, " | ".join(parts)))
    for line, text in rows[:38]:
        print(line, text[:250])

print("\n=== METADATA SAMPLE ===")
print(reference_meta.iloc[[0, 10, 100, 1000]][[
    "event", "hour", "last_value", "last_age_seconds",
    "last_z", "last_bin", "dt_seconds", "state_row"
]].to_string(index=False))
print("\n=== DYNAMIC IDENTIFICATION SUMMARY ===")
for title in ["## Sensor-contrast", "## Dynamic identification", "## Representation inspection"]:
    start = notes_text.find(title)
    if start >= 0:
        end = notes_text.find("\n## ", start + 4)
        segment = notes_text[start:end if end >= 0 else None]
        print(segment[:850])
print("\n=== BASELINES ===")
print("factual mean NLL:", factual_baseline.nll.mean())
print(dose_baseline.groupby("edit")["shift"].mean().to_string())
print("edits:", edited())
```
