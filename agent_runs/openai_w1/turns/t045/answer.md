The architecture confirms that drug inputs have a separate encoder, but their effects pass through shared state and prediction heads. An action-path edit could therefore change more than treatment responses. I’ll recover the cached regressions and helper functions next, looking for a correction tied to dose innovations; a useful candidate must improve rollouts without exceeding the factual-loss limit.

```python
import inspect
from pathlib import Path

_snapshot = list(globals().items())
_functions, _objects, _scalars = [], [], []
for name, value in sorted(_snapshot):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if callable(value):
        if getattr(value, "__module__", "") == "__main__":
            try:
                sig = str(inspect.signature(value))
            except Exception:
                sig = "(signature unavailable)"
            _functions.append(f"{name}{sig}")
    elif hasattr(value, "shape"):
        _objects.append(f"{name}: {type(value).__name__} {tuple(value.shape)}")
    elif isinstance(value, dict):
        _objects.append(f"{name}: dict[{len(value)}] keys={repr(list(value)[:8])[:160]}")
    elif isinstance(value, (list, tuple)):
        _objects.append(f"{name}: {type(value).__name__}[{len(value)}]")
    elif isinstance(value, (str, int, float, bool)):
        _scalars.append(f"{name}={repr(value)[:100]}")

print("=== RETAINED FUNCTIONS ===")
print("\n".join(_functions)[:2600])
print("\n=== RETAINED DATA / ANALYSIS OBJECTS ===")
print("\n".join(_objects)[:3400])
print("\n=== RETAINED SETTINGS ===")
print("\n".join(_scalars)[:1000])

print("\n=== VOCAB KEYS ===")
print(list(VOCAB))
for key in VOCAB:
    if any(s in str(key).lower() for s in ("config", "norm", "codes")):
        print(key, repr(VOCAB[key])[:1100])

print("\n=== POSTERIOR / ACTION / VALUE-HEAD TENSORS ===")
for name in NAMES:
    if name.startswith(("encode_action.", "z_post.", "z_prior.", "z_head.",
                        "last_numeric.", "query_norm.", "type_emb.")):
        print(name, SHAPES[name])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Restart, runs 36–38\n"
            "Run 36 reconstruction failed on a tuple/dict mismatch and retained substantial "
            "non-action posterior error; no edits. Run 37 re-read exact architecture. "
            "Drug encoder is separate but state and prediction heads are shared. "
            "Recovering cached objects before a reversible, forward-validated correction. "
            "Counts: 38 Python runs, 6 forward passes; no edits.\n")
```