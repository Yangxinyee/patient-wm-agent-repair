The architecture confirms that actions reach all prediction heads through the recurrent state; changing the posterior alone is not enough. The recovered estimates also disagree, especially for norepinephrine, so I should resolve that evidence before choosing a repair target. I’ll read the remaining identification results and locate the cached state-analysis helpers.

```python
import inspect

print("NOTEBOOK LENGTH:", len(notes_text))
print("\nIDENTIFICATION AND STATE ANALYSIS:")
print(notes_text[4300:10300])

print("\nSAVED ESTIMATE TABLES:")
for filename in ["dynamic_iv.csv", "sensor_iv.csv", "proxy_iv.csv", "kernel_targets.csv"]:
    tab = pd.read_csv("/work/" + filename)
    print(filename, tab.shape, list(tab.columns))
    print(tab.head(5).to_string(index=False))

print("\nAVAILABLE HELPERS:")
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if callable(obj) and getattr(obj, "__module__", None) == "__main__":
        try:
            print(name, str(inspect.signature(obj)))
        except Exception:
            print(name)

print("\nLARGE CACHED ARRAYS/TABLES:")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"tab", "obj"}:
        continue
    if isinstance(obj, (pd.DataFrame, np.ndarray)) and obj.size >= 10000:
        print(name, type(obj).__name__, obj.shape)

print("\nACTIVE EDITS:", edited())
```