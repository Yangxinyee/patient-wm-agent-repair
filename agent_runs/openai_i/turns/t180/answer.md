The architecture confirms that drug inputs enter the posterior before the recurrent update, so that remains a plausible repair pathway. Earlier record analyses support negative insulin effects and positive norepinephrine effects after history adjustment, but those one-hour estimates are not rollout targets.

The output was truncated before the helper inventory. I’ll retrieve that inventory and the remaining identification checks so the next candidate uses the established lag evidence rather than an assumed drug effect.

```python
from pathlib import Path
import inspect
import types

_notes = Path("/work/notes.md").read_text()
print("=== NOTEBOOK: IDENTIFICATION CHECKS ===")
print(_notes[4500:10500])

print("\n=== GLOBAL VARIABLE NAMES ===")
_names = [
    n for n, o in globals().copy().items()
    if not n.startswith("_") and not isinstance(o, types.ModuleType)
]
print(" ".join(sorted(_names))[:2400])

print("\n=== HELPER SIGNATURES ===")
_lines = []
for _n in sorted(_names):
    _o = globals()[_n]
    if inspect.isfunction(_o) and _n not in {"load", "load_original", "save_edit", "drop_edit", "edited"}:
        try:
            _lines.append(f"{_n}{inspect.signature(_o)}")
        except Exception:
            _lines.append(_n)
print("\n".join(_lines)[:1200])
print("\nNotebook characters:", len(_notes))
```
