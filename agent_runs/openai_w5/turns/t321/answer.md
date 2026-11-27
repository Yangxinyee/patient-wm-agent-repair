The recovered analysis supports a negative insulin effect and a positive norepinephrine effect, with persistence estimated under explicit causal assumptions. I still need the accumulated rollout targets and edit helpers. They will tell us whether shorter persistence could preserve the gain without breaking factual fit.

```python
import inspect, re

notes = Path("/work/notes.md").read_text()
nl = notes.splitlines()
print("NOTEBOOK LENGTH:", len(notes))
print("=== IDENTIFICATION, TARGETS, AND EDIT DESIGN ===")
selected = []
for i, line in enumerate(nl):
    if re.search(
        r"target|causal|identif|sensor-specific|phi|beta|neuron|private|pair|"
        r"coordinate|strength|held-out|cross.valid|negative control|dose.strat",
        line, re.I
    ):
        selected.append(f"{i}: {line}")
text = "\n".join(selected)
print(text[:5700])
print("=== GLOBAL NAMES ===")
names = [n for n in globals() if not n.startswith("_")]
print(", ".join(names)[:2300])
print("=== EDIT/FIT FUNCTIONS ===")
for n in names:
    ob = globals().get(n)
    if inspect.isfunction(ob) and re.search(
        r"edit|center|head|fit|build|pair|feature|action|restore|cache|transport", n, re.I
    ):
        try:
            print(n, inspect.signature(ob))
        except Exception:
            print(n)
print("=== KEY OBJECTS ===")
for n in ["BASE264", "BASE275", "W", "WO", "pairs", "panel", "panel_tune"]:
    if n in globals():
        ob = globals()[n]
        print(n, type(ob).__name__, getattr(ob, "shape", None),
              list(ob)[:8] if isinstance(ob, dict) else "")
```
