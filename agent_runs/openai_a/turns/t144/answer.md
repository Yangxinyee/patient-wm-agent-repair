The recovered notes support an insulin one-hour effect near −1.93 per rate unit, but norepinephrine estimates conflict and its proposed instrument is weak. I won’t treat those NE estimates as causal. Next I’ll inspect the recurrent-update implementation; correcting it should remove the large state errors without changing any weights.

```python
import dis
import numpy as np

print("=== VOCAB ENCODING CONFIGURATION ===")
print("keys:", list(VOCAB))
for key in VOCAB:
    if any(s in key.lower() for s in ("categor", "unit", "kind", "normal")):
        print(key, repr(VOCAB[key])[:1200])
print("codes:", VOCAB.get("codes"))
print("rm_states fields:", list(rm_states) if isinstance(rm_states, dict) else type(rm_states))
print("rm_normspec:", repr(globals().get("rm_normspec"))[:700])

print("\n=== COMPACT RECURRENT-STEP IMPLEMENTATION ===")
skip_ops = {"RESUME", "CACHE", "NOP", "PRECALL", "PUSH_NULL", "COPY_FREE_VARS"}
for name in ["rec_step_with_z", "rec_memory"]:
    print("\nFUNCTION", name)
    func = globals()[name]
    lines = [
        f"{i.offset}: {i.opname} {i.argrepr}"
        for i in dis.get_instructions(func)
        if i.opname not in skip_ops
    ]
    print("\n".join(lines)[:3500])

print("\n=== BEST PREVIOUS TRANSITION CONFIGURATIONS ===")
obj = globals().get("rec_transition_checks")
if isinstance(obj, list):
    try:
        print(sorted(obj)[:8])
    except (TypeError, ValueError):
        print(repr(obj[:8])[:1100])
else:
    print(repr(obj)[:1100])

print("\n=== LAB-SPECIFIC NORMALIZATION ===")
for key in ("normalization", "normalisation", "norm", "norms"):
    obj = VOCAB.get(key)
    if isinstance(obj, dict):
        print(key, {k: v for k, v in obj.items() if "glucose" in str(k)})
```
