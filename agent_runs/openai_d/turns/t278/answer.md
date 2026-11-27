The last result separated the NE correction’s dose-sensitive component from its reading-type gate. I’ll recover the exact head computation before testing factual cancellation using existing hidden features. Success requires better held-out factual loss **and** preserved actual rollout effects; approximate probe preservation alone is insufficient.

```python
from pathlib import Path
import inspect, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RELEVANT FUNCTIONS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and name.startswith(("c210_", "r230_", "r231_", "r232_", "r236_", "r237_")):
        if hasattr(obj, "__code__"):
            co = obj.__code__
            print(name, "args:", co.co_varnames[:co.co_argcount], "uses:", co.co_names)
            try:
                src = inspect.getsource(obj)
                if len(src) <= 5500:
                    print(src)
            except (OSError, TypeError):
                pass

print("\n=== COMPACT CANDIDATE OBJECTS ===")
for name in sorted(list(globals())):
    if name.startswith(("r232_", "r235_", "r237_", "r231_")):
        obj = globals()[name]
        if isinstance(obj, np.ndarray):
            print(name, "array", obj.shape, obj.dtype)
        elif isinstance(obj, dict):
            print(name, "dict", [
                (str(k), tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                for k, v in list(obj.items())[:12]
            ])
        elif isinstance(obj, (int, float, str, tuple, list)):
            print(name, repr(obj)[:250])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 238: restart and cancellation audit\n"
            "Retained validated safe edit 6288c226d6. Re-read architecture and recovered "
            "candidate/evaluator interfaces before testing training-only nonlinear "
            "factual cancellation. No weights changed.\n")
```