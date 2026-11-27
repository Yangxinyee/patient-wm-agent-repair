The estimated responses persist for several hours, so matching only the next-hour slope would miss much of the dose effect. I’ll locate the resulting rollout targets and the support-guard parameters. No weights will change in this step.

```python
import inspect

print("=== EVIDENCE AFTER THE IMPULSE ESTIMATES ===")
pos = notes_text.find("## Run 6")
print(notes_text[pos:pos + 3500] if pos >= 0 else "Run 6 marker not found")

print("\n=== CURRENT RELEVANT VARIABLES ===")
prefixes = ("r159", "r157", "r115", "r139", "r138", "r149", "r156")
for name in sorted(k for k in list(globals()) if k.startswith(prefixes)):
    obj = globals()[name]
    if isinstance(obj, dict):
        description = f"dict keys={list(obj)[:18]}"
    elif hasattr(obj, "shape"):
        description = f"shape={tuple(obj.shape)}"
    elif isinstance(obj, (list, tuple)):
        description = f"{type(obj).__name__}[{len(obj)}]"
    elif callable(obj):
        description = str(inspect.signature(obj))
    else:
        description = repr(obj)[:130]
    print(name, description)

print("\n=== SUPPORT-GUARD HELPER ===")
try:
    print(inspect.getsource(r159_pair))
except (OSError, TypeError):
    import dis
    dis.dis(r159_pair)
```
