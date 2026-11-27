The helper uses a manually assembled register, but the output was truncated before its construction appeared. I’ll retrieve just those inputs and the key replay objects. If the remaining error is mostly prior-sampling noise, averaging many prior draws should reduce it; if it persists, the register is still wrong.

```python
import dis, re
g125 = dict(globals())
print("=== RELEVANT GLOBAL NAMES ===")
for suffix125 in (97, 103, 104, 105, 118, 120, 121):
    print(suffix125, [k for k in sorted(g125) if re.search(str(suffix125)+r"$", k)])
print("=== REGISTER CANDIDATES ===")
for name125 in ("attrs120", "attrs121", "reg97", "regs97", "last97",
                "register97", "codes97", "sel97", "attr_static105", "attrbest121"):
    if name125 in g125:
        obj125 = g125[name125]
        if hasattr(obj125, "shape"):
            print(name125, tuple(obj125.shape), str(obj125[:3])[:650])
        else:
            print(name125, str(obj125[:2])[:650])
print("=== COMPACT REPLAY BYTECODE ===")
for name125 in ("next_view105", "state_update82", "isolated_record", "memory82"):
    f125 = g125[name125]
    instructions125 = []
    for ins125 in dis.get_instructions(f125):
        if ins125.opname not in ("RESUME", "CACHE", "PRECALL"):
            instructions125.append(ins125.opname + ":" + ins125.argrepr)
    print(name125, " | ".join(instructions125)[:2100])
print("=== VOCAB CONFIG AND CODES ===")
print(VOCAB["config"])
print(VOCAB["codes"])
print("=== LAST_NUMERIC COLUMN NORMS ===")
print(torch.linalg.vector_norm(load_original("last_numeric.weight"), dim=0))
```