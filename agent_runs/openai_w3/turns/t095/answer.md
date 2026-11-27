The architecture confirms that drug inputs can strongly affect the recurrent state through the prelude, unlike the ineffective latent-only edit. No weights have changed. I’ll locate the retained replay functions and causal-effect estimates so the next edit targets that active path. The hypothesis would be supported if it changes dose responses while leaving recorded-dose predictions nearly unchanged.

```python
import inspect
from pathlib import Path

print("LOCAL FUNCTIONS")
for k in sorted(list(globals())):
    v = globals()[k]
    if callable(v) and getattr(v, "__module__", "") == "__main__" and hasattr(v, "__code__"):
        c = v.__code__
        print(k, c.co_varnames[:c.co_argcount], "defaults", str(getattr(v, "__defaults__", None))[:100])

print("\nRECENT REPLAY VARIABLES")
for k in sorted(list(globals())):
    if k.startswith(("r81", "r82", "replay", "state", "s81", "s82", "W81", "W82")):
        v = globals()[k]
        print(k, type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else
              list(v)[:12] if isinstance(v, dict) else str(v)[:180])

print("\nCORE TENSORS")
for n in NAMES:
    if n.startswith(("core.prelude.", "core.blocks.", "core.coda.", "core.b_in", "core.c_out", "core.delta", "core.a_in", "core.flow")):
        print(n, SHAPES[n])

print("\nEARLY NOTEBOOK EVIDENCE")
print(Path("/work/notes.md").read_text()[:5500])
print("\nCURRENT EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 84; forwards 7. Architecture confirms a non-negligible action-token prelude path. Recovering exact replay functions and record-only identification evidence before constructing a factual-policy-residual correction. No edits.\n")
```