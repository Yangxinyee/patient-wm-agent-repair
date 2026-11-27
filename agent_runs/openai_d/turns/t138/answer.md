The records support immediate residual-dose slopes near −1.99 for insulin and +3.29 for norepinephrine. The longer-horizon targets are less certain because they require separating persistence from later treatment feedback. I’ll recover the saved construction and analysis helpers next; any new candidate must pass both paired factual and rollout checks.

```python
import inspect
print("=== KERNEL FUNCTIONS ===")
for _name, _obj in sorted(list(globals().items())):
    if inspect.isfunction(_obj) and not _name.startswith("_"):
        try:
            print(_name, str(inspect.signature(_obj))[:180])
        except Exception:
            print(_name)
print("=== RELEVANT OBJECTS ===")
for _name, _obj in sorted(list(globals().items())):
    if any(s in _name.lower() for s in
           ["kernel", "causal", "dml", "innovation", "ne_", "norepi",
            "r115", "r106", "r104", "r99", "gate", "coef", "strength"]):
        print(_name, type(_obj).__name__, getattr(_obj, "shape", ""))
print("=== NOTES: LONG-HORIZON ESTIMATES ===")
_nt = Path("/work/notes.md").read_text()
_i = _nt.find("## Run 5")
_j = _nt.find("## Run 9", _i + 1)
print(_nt[_i:(_j if _j >= 0 else _i + 6000)][:5200])
```
