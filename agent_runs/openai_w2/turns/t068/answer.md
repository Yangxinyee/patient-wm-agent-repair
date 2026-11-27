All edits are now reverted. The kernel contains earlier head-reconstruction and dose-probe work, so I’ll review those results before repeating any experiment. A head-only edit is worth reconsidering only if that work found a dose-sensitive direction with little factual error.

```python
import re
text = Path("/work/notes.md").read_text()
sections = re.split(r"(?=^## )", text, flags=re.M)
wanted = ["Run 16:", "Run 17:", "Run 18:", "Run 22:", "Run 24:",
          "Runs 25", "Run 27:", "Run 30:", "Run 31:", "Run 32:", "Run 33:"]
for sec in sections:
    if any(w in sec.splitlines()[0] for w in wanted):
        print(sec)
print("\nALL USER FUNCTIONS:")
for key, val in list(globals().items()):
    if callable(val) and getattr(val, "__module__", "") == "__main__":
        print(key)
print("\nSTATE / HEAD METADATA:")
print(state_meta.head().to_string(index=False))
print(head_data.head(3).to_string(index=False))
print("\nSTATE-RELATED OBJECTS:")
for key, val in list(globals().items()):
    if any(s in key.lower() for s in ["probe", "design", "state", "ridge", "head", "policy", "backup"]):
        if isinstance(val, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(key, type(val).__name__, tuple(val.shape))
        elif isinstance(val, dict):
            print(key, "dict keys", list(val.keys())[:12])
```
