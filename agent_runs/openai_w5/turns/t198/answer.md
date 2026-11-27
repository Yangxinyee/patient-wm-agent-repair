The helper inventory shows that attention-layout and memory-handling choices remain unresolved in the emulator. I’ll inspect the actual step calculation and recent transport diagnostics without the large cached objects. If those choices explain the state mismatch, we can correct it; otherwise, we’ll need a more direct dose-channel test.

```python
import dis, re, collections

def compact_restart171(f, maxchars=6500):
    lines = collections.OrderedDict()
    line = f.__code__.co_firstlineno
    for ins in dis.get_instructions(f):
        if ins.starts_line is not None:
            line = ins.starts_line
        if ins.opname in {"RESUME", "CACHE", "PRECALL", "NOP", "COPY", "SWAP", "PUSH_NULL", "EXTENDED_ARG"}:
            continue
        arg = ins.argrepr
        if ins.opname == "LOAD_CONST" and len(arg) > 150:
            arg = arg[:150] + "..."
        lines.setdefault(line, []).append(ins.opname + (" " + arg if arg else ""))
    text = "\n".join(str(k) + ": " + "; ".join(v) for k, v in lines.items())
    print(text[:maxchars])

print("=== STATE-STEP IMPLEMENTATION ===")
compact_restart171(wm_step_saved, 6600)
print("\n=== SET ENCODER ===")
compact_restart171(wm_set, 1300)

print("\n=== STATE DATASETS ===")
for k, v in sorted(list(globals().items())):
    if isinstance(v, dict) and {"x", "stay_id", "hour"}.issubset(v):
        print(k, v["x"].shape, "stays", len(np.unique(v["stay_id"])))

print("\n=== RECENT DIAGNOSTIC OBJECTS ===")
for k, v in sorted(list(globals().items())):
    match = re.search(r"(\d+)$", k)
    if not match or not 162 <= int(match.group(1)) <= 170:
        continue
    if callable(v):
        continue
    if hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", len(v), list(v)[:8])
    else:
        print(k, str(v)[:250])

print("\n=== TRANSPORT TRIALS ===")
for k in ("wm_transition_trials", "wm_transition_trials2", "best67", "best82", "choices163"):
    if k in globals():
        print(k, str(globals()[k])[:1300])
```
