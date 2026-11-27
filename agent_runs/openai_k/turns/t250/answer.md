```python
from pathlib import Path
import inspect, dis, io, types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("=== RELEVANT TENSOR SHAPES ===")
for n in NAMES:
    if any(s in n for s in ("core.", "out", "head", "value", "z_in", "z_post")):
        print(n, SHAPES[n])

print("=== REPLAY FUNCTION DEFINITIONS ===")
for name in ("repaired_core_local", "local_token154", "r59_core",
             "r179_capture_head", "rr35_scaled_action_encoding"):
    f = globals().get(name)
    if not callable(f):
        continue
    print("\nFUNCTION", name)
    try:
        print(inspect.getsource(f)[:7000])
    except (OSError, TypeError):
        buf = io.StringIO()
        dis.dis(f, file=buf)
        print(buf.getvalue()[:7000])

print("=== RETAINED INTERFACE METADATA ===")
for name in sorted(list(globals())):
    if any(s in name.lower() for s in
           ("candidate158", "context151", "physical", "causal", "r49_", "backup218")):
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, {str(k): (tuple(v.shape) if hasattr(v, "shape")
                                  else type(v).__name__) for k, v in obj.items()})
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, (str, int, float, list, tuple)):
            print(name, repr(obj)[:1200])
        else:
            print(name, type(obj).__name__)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 225: authoritative replay recovery\n"
            "- Run 224 failed through tuple indexing in the local diagnostic; no weights changed.\n"
            "- Current six-tensor candidate preserves factual fit but fails causal rollout.\n"
            "- Recovering exact architecture and replay definitions before choosing a different edit boundary.\n"
            "- Python runs: 225; forward passes: 17.\n")
```