The records support multi-hour effects: insulin’s impulse peaks around the second hour, while norepinephrine’s is shorter-lived. I also found a likely reconstruction bug: the current atom helper appears to encode drug numbers but not measurement values. I’ll check that explicitly and fix the input reconstruction; close posterior matches for measurements and infusion endings would confirm the fix.

```python
import dis

print("=== VOCABULARY STRUCTURE ===")
print("keys:", list(VOCAB))
for k in VOCAB:
    if k not in {"config", "numeric_specs", "value_bin_centres", "rate_bin_centres",
                 "dose_bin_centres", "numeric"}:
        text = repr(VOCAB[k])
        if len(text) < 1400:
            print(k, text)
print("embed shapes:", {n: SHAPES[n] for n in NAMES
                        if n.startswith("embed.")})
print("gap prefix:", globals().get("gap_prefix41"))
print("category globals:", [(n, repr(v)[:600]) for n, v in list(globals().items())
                            if "categor" in n and isinstance(v, (dict, list, tuple))])

def compact_instructions(fn):
    lines = {}
    current = None
    for ins in dis.get_instructions(fn):
        if ins.starts_line is not None:
            current = ins.starts_line
        if ins.opname not in {"RESUME", "CACHE", "NOP"}:
            lines.setdefault(current, []).append(ins.opname + " " + ins.argrepr)
    for line, entries in lines.items():
        print(line, " | ".join(entries))

print("\n=== ATOM BASE IMPLEMENTATION ===")
compact_instructions(repair_atom_base)
print("\n=== START CATEGORIES IMPLEMENTATION ===")
compact_instructions(latent_start_atom)
print("\n=== CATEGORY IMPLEMENTATION ===")
compact_instructions(latent_categories)
print("\n=== CONTEXT IMPLEMENTATION ===")
compact_instructions(latent_context)
```
