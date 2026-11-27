The recovered notes show large, history-adjusted dose effects in the records, unlike the original model’s near-zero rollout responses. The output was truncated before the retained helpers appeared. I’ll retrieve those compactly next; this will let us test whether the proposed transport carries the measured dose changes without disturbing factual predictions.

```python
import inspect, re
notes189 = Path("/work/notes.md").read_text()
print("=== RECORD-BASED ROLLOUT TARGETS ===")
for marker in ["Python runs: 10;", "Python runs: 11;", "Python runs: 12;"]:
    p = notes189.find(marker)
    if p >= 0:
        print(notes189[p:notes189.find("\n\n", p)][:1200])

print("\n=== RECENT OBJECT INVENTORY ===")
snap189 = list(globals().items())
lines189 = []
for name, obj in snap189:
    if name.startswith("_") or not re.search(r"(154|169|186|187)$", name):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        lines189.append(f"{name}: {tuple(obj.shape)} {obj.dtype}")
    elif isinstance(obj, dict):
        lines189.append(f"{name}: dict {len(obj)} keys={list(obj)[:8]}")
    elif isinstance(obj, (list, tuple)):
        lines189.append(f"{name}: {type(obj).__name__} {len(obj)}")
    elif isinstance(obj, (int, float, bool, str)):
        lines189.append(f"{name}: {repr(obj)[:70]}")
print("\n".join(lines189)[:3300])

print("\n=== LOCAL FUNCTION SIGNATURES ===")
lines189 = []
for name, obj in snap189:
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            lines189.append(f"{name}{inspect.signature(obj)}")
        except Exception:
            lines189.append(name)
print("\n".join(lines189)[:3000])
print("\nHistory available:", {k: type(globals()[k]).__name__
      for k in ["_ih", "In", "get_ipython"] if k in globals()})
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 189; forwards 10. Architecture and schema recovered. Prior record-only impulse estimates show substantial insulin and NE effects; original rollout is near-zero/wrong-sign. Run 188 output truncated before helper inventory, so retrieving compact targets and retained objects. No weight changes.\n")
```
