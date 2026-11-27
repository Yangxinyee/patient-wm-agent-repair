The architecture confirms that dose information enters the recurrent state, while predictions also depend on last-reading registers. Thus a next-hour effect alone cannot determine the rollout effect. The output was truncated before the helper inventory; I’ll retrieve that compactly so I can reuse the existing reconstruction and test persistence without changing weights.

```python
snapshot71 = list(globals().items())
helper_lines71 = []
object_lines71 = []
for name, obj in snapshot71:
    if name.startswith("_"):
        continue
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            helper_lines71.append(name + str(inspect.signature(obj)))
        except Exception:
            helper_lines71.append(name)
        continue
    if not re.search(r"(causal|interval|panel|lag|norep|(^|_)ne($|_)|direction|candidate|contrast|feature|valid|target|audit|policy|state|bundle|fit69|sample|pred)", name, re.I):
        continue
    if isinstance(obj, dict):
        object_lines71.append(f"{name}: dict {list(obj)[:9]}")
    elif hasattr(obj, "shape"):
        object_lines71.append(f"{name}: {type(obj).__name__} {tuple(obj.shape)}")
    elif isinstance(obj, (list, tuple)):
        object_lines71.append(f"{name}: {type(obj).__name__}[{len(obj)}] {repr(obj[:3])[:100]}")
    elif isinstance(obj, (int, float, str)):
        object_lines71.append(f"{name}: {repr(obj)[:100]}")
print("HELPERS\n" + "\n".join(helper_lines71)[:4300])
print("\nOBJECTS\n" + "\n".join(object_lines71)[:4200])
print("\nDATA DEFINITIONS\n" + Path("/data/README.md").read_text()[:600])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 71: bounded helper inventory\n"
            "Run70 output was truncated before the inventory. Retrieved bounded helper/object "
            "summaries to reuse the established local reconstruction. No weights changed. "
            "Python runs: 71; forwards: 10.\n")
```
