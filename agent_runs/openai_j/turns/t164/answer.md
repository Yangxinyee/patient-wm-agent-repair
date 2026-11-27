The architecture confirms that changing the value head leaves the other prediction heads unchanged directly, whereas changing the recurrent state would affect them all. The earlier record estimates also show why simple dose associations are unreliable. I’ll now inspect the cached stochastic features and the later causal checks. I’m looking for stable drug-specific estimates and features that can support a correction without the latent-noise sensitivity that broke earlier candidates.

```python
import inspect
print("=== FUNCTIONS ===")
for name, obj in list(globals().items()):
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, str(inspect.signature(obj))[:220])
        except Exception:
            pass

print("\n=== DATAFRAMES ===")
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame):
        print(name, obj.shape, ",".join(map(str, obj.columns[:12])))

print("\n=== RELEVANT DICTIONARIES ===")
for name, obj in list(globals().items()):
    if isinstance(obj, dict) and (
        any(s in name.lower() for s in ["contrast", "head", "state", "fit", "meta", "coef", "mc", "stoch", "cache"])
        or ("insulin", 2.0) in obj
    ):
        print(name, "keys:", str(list(obj.keys())[:12])[:350])
        if ("insulin", 2.0) in obj:
            g = obj[("insulin", 2.0)]
            if isinstance(g, dict):
                for k, v in g.items():
                    print(" ", k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)

print("\n=== NOTEBOOK HEADINGS ===")
print("\n".join(line for line in notes_text.splitlines() if line.startswith("##")))

print("\n=== CAUSAL RESULT FILES ===")
for filename in ["dynamic_iv.csv", "sensor_iv.csv", "proxy_iv.csv", "kernel_targets.csv"]:
    df = pd.read_csv("/work/" + filename)
    print(filename, df.shape, "\n", df.head(6).to_string(index=False))
print("\nActive edits:", edited())
```
