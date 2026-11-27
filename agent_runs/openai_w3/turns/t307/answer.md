The baseline is not simply at a glucose ceiling: insulin halving raises glucose in only 42% of reached rows, and the repaired baseline drifts upward. Saved tests also found roughly symmetric local dose signals. I’ll examine the recurrent update and replay helpers next; sustained-rollout feedback is now a stronger suspect than dose decoding.

```python
import inspect, dis, io, numpy as np
from pathlib import Path

print("=== REPLAY HELPER INTERFACES ===")
for name in ["posterior153", "latent206", "prelude81", "prelude155", "parts254",
             "state_update82", "memory82", "next_view105", "view171", "phys171",
             "response186", "response_raw234", "raw_hidden228", "hidden220"]:
    fn = globals().get(name)
    if callable(fn) and hasattr(fn, "__code__"):
        print(name, str(inspect.signature(fn)), "names:", fn.__code__.co_names,
              "constants:", [x for x in fn.__code__.co_consts
                              if isinstance(x, (str, int, float, tuple))][:18])

print("=== STATE UPDATE IMPLEMENTATION ===")
buf272 = io.StringIO()
dis.dis(state_update82, file=buf272)
print(buf272.getvalue()[:3600])
print("=== SENSOR READOUT IMPLEMENTATION ===")
buf272 = io.StringIO()
dis.dis(phys171, file=buf272)
print(buf272.getvalue()[:1900])

print("=== OUTPUT / RECURRENCE TENSOR SHAPES ===")
for name in NAMES:
    if any(s in name for s in ["attr_heads.value", "last_numeric", "core.flow",
                                "core.delta_in", "core.a_in", "core.b_in",
                                "z_head", "z_prior"]):
        print(name, SHAPES[name])

print("=== RETAINED REPLAY INPUTS ===")
for name, obj in list(globals().items()):
    if re.search(r"^(D|C|U|E|H|T|M|ctx|rows|states|arrays|meta|mem|h)_?\d*$|81$|82$|185$|192$|193$|206$", name):
        if isinstance(obj, dict):
            print(name, "dict keys", list(obj)[:14])
        elif hasattr(obj, "shape"):
            print(name, tuple(obj.shape))
        elif isinstance(obj, (list, tuple)):
            print(name, "length", len(obj))

with Path("/work/notes.md").open("a") as f:
    f.write("\nPython run 272; forwards 22. No simple glucose ceiling: half-dose positives only 41.6%; repaired baseline drifts glucose +~10 and MAP -~3 units. Saved carrier tests are approximately symmetric. Recovering recurrence/readout replay to address downstream feedback rather than repeating scalar payload scaling. No edits changed.\n")
```
