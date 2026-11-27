The earlier head reconstruction did not match official predictions closely enough to trust its absolute NLL. I’ll check what was already fitted and which state-query helpers are available, keeping the output compact. If a reliable query reconstruction exists, I can isolate dose information without relying on the uncertain last-reading layout; otherwise that route needs validation first.

```python
import inspect
from pathlib import Path

print("=== FUNCTIONS: COMPACT ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and not name.startswith("_"):
        print(name, str(inspect.signature(obj)))

print("\n=== NOTES 115–119 ===")
for s in Path("/work/notes.md").read_text().split("\n## "):
    if any(s.startswith("Run " + str(k)) or s.startswith("Runs " + str(k))
           for k in range(115, 120)):
        print("\n## " + s[:1000])

print("\n=== RELEVANT OBJECT SUMMARIES ===")
for name in ("features119", "proj118", "numproj118", "fit119", "result119",
             "models145", "diagnostics145", "readoutweights151",
             "contexts151", "contexts157", "jac157", "directions158",
             "r49_view", "V144o", "V144e", "W0"):
    if name not in globals():
        continue
    obj = globals()[name]
    if isinstance(obj, dict):
        print(name, "keys:", list(obj)[:16])
    elif hasattr(obj, "shape"):
        print(name, "shape:", obj.shape)
    else:
        print(name, type(obj).__name__, str(obj)[:250])

print("\n=== QUERY/UPDATE HELPER DEPENDENCIES ===")
for name in ("r57_features", "r59_features", "audit_core_update",
             "build_candidate89", "assay_head110"):
    fn = globals().get(name)
    if inspect.isfunction(fn):
        print(name, "names:", fn.__code__.co_names)
        print("constants:", [c for c in fn.__code__.co_consts
                             if isinstance(c, (str, int, float))][:35])

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 162: audit readout validation before redesign\n"
            "- Earlier absolute-head reconstruction was not validated; do not treat its NLL as authoritative.\n"
            "- Inspecting compact helper interfaces for a register-independent dose decoder and explicit value-head correction.\n"
            "- No weights changed. Python runs: 162; forward passes: 10.\n")
```
