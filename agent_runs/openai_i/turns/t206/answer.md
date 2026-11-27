The architecture confirms that the value head reads drug effects only after they pass through the recurrent state. The local regression is therefore only a proxy for rollout behavior. I’ll recover its candidate weights and validation helpers, then test the smallest correction on the actual forward tool. It is promising only if it improves rollouts while meeting the factual-fit limit.

```python
import re, inspect

print("--- objects from the latest calibration ---")
latest185 = []
for key, obj in list(globals().items()):
    if re.search(r"(181|182|183)$", key):
        latest185.append(key)
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(key, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(key, "dict", len(obj), "keys:", list(obj)[:16])
        elif isinstance(obj, (list, tuple)):
            print(key, type(obj).__name__, len(obj), repr(obj[:5])[:350])
        elif callable(obj):
            print(key, "function", inspect.signature(obj),
                  "references:", obj.__code__.co_names)
        else:
            print(key, type(obj).__name__, repr(obj)[:250])

print("\n--- relevant helper signatures ---")
for key, obj in list(globals().items()):
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        if any(s in key.lower() for s in ["head", "read", "install", "fit", "feature", "register"]):
            print(key, inspect.signature(obj))

print("\n--- value-head original weights ---")
for name in ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
             "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"]:
    a = load_original(name)
    print(name, tuple(a.shape), "RMS", float(np.sqrt(np.mean(np.asarray(a)**2))))

print("\n--- original tuning factual summaries available ---")
for path in sorted(Path("/work").glob("fwd_*/factual_original.csv")):
    print(str(path))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 185 — recover smallest direct-head candidate\n"
            "- Architecture confirms a shared value head downstream of action-dependent recurrence.\n"
            "- Recovering alpha-1 regression and helper contracts for an actual GPU test; proxy fit alone is not certification.\n")
```