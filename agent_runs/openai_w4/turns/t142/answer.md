The architecture confirms that the dose channel enters a recurrent state, where a small one-step change can amplify later. The diagnostic output was truncated before the useful details. I’ll recover those details compactly, including the record-based effect estimates, before choosing a stability constraint.

```python
import inspect
parts127 = []
for name in ["replay_factual125", "transition73", "transition_with_b110", "probe_means95"]:
    fn = globals().get(name)
    if callable(fn):
        parts127.append(f"\n{name}{inspect.signature(fn)}")
        try:
            parts127.append(inspect.getsource(fn)[:3500])
        except (OSError, TypeError):
            parts127.append("locals=" + repr(fn.__code__.co_varnames))
            parts127.append("globals=" + repr(fn.__code__.co_names))
            parts127.append("constants=" + repr(fn.__code__.co_consts)[:1000])
print("\n".join(parts127)[:6500])
print("\n--- Compact retained diagnostics ---")
for name, obj in list(globals().items()):
    if any(s in name for s in ["118", "125", "110"]) and not callable(obj):
        if isinstance(obj, dict):
            print(name, [(k, getattr(v, "shape", type(v).__name__))
                         for k, v in obj.items() if k != "weights"][:14])
        elif hasattr(obj, "shape"):
            print(name, obj.shape)
print("\n--- Initial record evidence ---")
print(open("/work/notes.md").read()[:4500])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 127\nArchitecture confirms that the repair enters a recurrent pathway. Recovering compact replay interfaces and initial causal evidence; the previous diagnostic output was truncated. No additional weights edited.\n")
```