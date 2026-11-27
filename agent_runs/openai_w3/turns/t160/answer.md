The records support substantial insulin lowering and a positive norepinephrine effect, unlike the original rollout. The architecture offers a direct action-to-state pathway, but the failed pilot did not calibrate it reliably. I will recover the exact replay helpers and dose decoder, then test whether an explicit dose change reaches the next prediction.

```python
import inspect, linecache

print("RELEVANT TENSORS")
for n144 in NAMES:
    if any(s144 in n144 for s144 in (
        "attr_heads.value", "core.prelude", "core.coda", "core.c_out",
        "core.b_in", "core.a_in", "core.delta_in", "z_post", "z_prior"
    )):
        print(n144, SHAPES[n144])

print("\nEARLIER EXPERIMENTS, RUNS 11–76")
pars144 = notes143.split("\n\n")
sel144 = []
for p144 in pars144:
    import re
    m144 = re.search(r"Python runs?:\s*(\d+)|Python run\s+(\d+)", p144)
    if m144:
        number144 = int(next(v144 for v144 in m144.groups() if v144))
        if 11 <= number144 <= 76:
            sel144.append(p144)
print("\n\n".join(sel144)[-5400:])

print("\nRETAINED HELPERS")
for name144 in (
    "encode_groups80", "prelude81", "prelude_cache88", "state_update82",
    "parts105", "next_view105", "means105"
):
    fn144 = globals().get(name144)
    if callable(fn144):
        print("\n", name144, inspect.signature(fn144))
        try:
            src144 = inspect.getsource(fn144)
        except Exception:
            src144 = "".join(linecache.getlines(fn144.__code__.co_filename))
            if not src144:
                src144 = "Referenced globals: " + repr(fn144.__code__.co_names)
        print(src144[:1900])

print("\nRECENT PILOT OBJECTS")
for name144 in sorted(list(globals())):
    if name144.endswith(("133", "134", "135", "136")):
        obj144 = globals()[name144]
        if isinstance(obj144, (np.ndarray, torch.Tensor)):
            print(name144, tuple(obj144.shape))
        elif callable(obj144):
            print(name144, inspect.signature(obj144))
        elif isinstance(obj144, dict):
            print(name144, "keys", list(obj144)[:8])

with open("/work/notes.md", "a") as f144:
    f144.write("\nPython run 144; forwards 9. Record-derived insulin and NE effects disagree strongly with original rollouts. Recovering exact action/prelude/core helpers and explicit dose decoder to validate transport before another edit. No weights changed.\n")
```