The architecture confirms that the current edit changes only the reading-value head. The records support opposite immediate effects for the two drugs, but those slopes are not the full rollout targets. I’ll recover the longer-horizon estimates and cached edit variables before calibrating another candidate.

```python
start = notes_text.find("## Run 5")
print("=== LONGER-HORIZON EVIDENCE ===")
print(notes_text[start:start + 4700])

print("\n=== RECENT EDIT VARIABLES ===")
selected = [
    line for line in inventory
    if line.startswith(("r159", "r158", "r157", "r115", "r139", "r149"))
]
print("\n".join(selected)[:4300])

print("\n=== AVAILABLE HELPER FUNCTIONS ===")
import types
print(", ".join(sorted(
    k for k, v in list(globals().items())
    if isinstance(v, types.FunctionType) and not k.startswith("_")
)))
```
