The earlier checks found dose-threshold effects, but also changes in which readings were recorded; those selection effects matter. The tensor layout also differs from the abbreviated architecture description: `last_numeric` takes eight inputs. I’ll inspect the existing reconstruction and the independent effect checks before choosing an edit.

```python
import inspect
import dis
import io
from pathlib import Path

notes = Path("/work/notes.md").read_text()
for title in [
    "## Run 6:", "## Run 10:", "## Run 11:", "## Run 12:"
]:
    start = notes.find(title)
    if start >= 0:
        end = notes.find("\n## ", start + 1)
        section = notes[start:end if end >= 0 else None]
        print(section[:1600])

print("\n=== RECONSTRUCTION CONFIGURATION ===")
print(head_cfg)
print("\n=== FUNCTION SIGNATURE ===")
print(inspect.signature(local_value_head))

print("\n=== RECONSTRUCTION SOURCE ===")
try:
    source = inspect.getsource(local_value_head)
except (OSError, TypeError):
    buf = io.StringIO()
    dis.dis(local_value_head, file=buf)
    source = buf.getvalue()
print(source[:4700])

print("\n=== HEAD DATA ===")
print(head_data.columns.tolist())
print(head_data.head(2).to_string(index=False))

print("\n=== STATE OBJECTS ===")
for key, value in list(globals().items()):
    if any(term in key.lower() for term in ["state", "npz", "stays"]):
        print(key, type(value).__name__, getattr(value, "shape", ""),
              list(value.keys())[:10] if isinstance(value, dict) else "")

print("\n=== HEAD WEIGHT KEYS ===")
print(list(head_weights))
```