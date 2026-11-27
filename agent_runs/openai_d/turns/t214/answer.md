The stronger edit brings norepinephrine doubling to +4.736, close to the provisional record-derived estimate, but halving remains too weak at −1.201. Insulin responses are largely unchanged. I’ll recover the factual result and the evidence behind the target effects before adjusting further. A factual pass and evidence for proportional dose effects would support addressing this asymmetry.

```python
from pathlib import Path
import json, numpy as np

print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
print("\nRECORD SCHEMA\n", Path("/data/README.md").read_text())
print("\nFORWARD 24 FILES")
for p in sorted(Path("/work/fwd_24").iterdir()):
    print(p.name, p.stat().st_size)
    if p.suffix in (".json", ".txt"):
        print(p.read_text()[:16000])

notes184 = Path("/work/notes.md").read_text()
print("\nEARLY NOTEBOOK\n", notes184[:24000])
print("\nRECENT KERNEL OBJECTS")
for key in sorted(list(globals())):
    if key.startswith(("c17", "c18", "r17", "r18", "r165", "r170", "r171")):
        obj = globals()[key]
        if isinstance(obj, dict):
            print(key, "dict keys:", list(obj)[:35])
        elif hasattr(obj, "shape"):
            print(key, type(obj).__name__, obj.shape)
        elif isinstance(obj, (list, tuple)):
            print(key, type(obj).__name__, "length", len(obj))
        elif isinstance(obj, (str, int, float)):
            print(key, repr(obj)[:300])
print("\nOTHER NAMED OBJECTS\n", [
    k for k in list(globals())
    if not k.startswith("_") and not k.startswith(("r", "c"))
][-180:])
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 184: restart audit\nForward 25 stronger-NE trial: insulin -14.423/+6.644; NE +4.736/-1.201 on tuning 0..99. Doubling matches the provisional cohort target but half-dose remains weak. No additional edit; recovering factual result and causal-estimation evidence before further changes.\n")
```
