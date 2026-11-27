The factual test passed: loss increased by 0.00247 nats, below the 0.005 limit. But the rollout barely changed, so factual safety alone does not justify keeping this candidate. I will inspect the cached update functions and earlier identification evidence to find where the dose signal is being lost.

```python
import re

print("=== IDENTIFICATION NOTES (EARLY) ===")
_text254 = open("/work/notes.md").read()
print(_text254[:5500])

print("\n=== RECENT OBJECTS ===")
_recent254 = [(k, v) for k, v in list(globals().items())
              if re.search(r"(23[6-9]|24[0-9]|25[0-2])", k) and not k.startswith("_")]
for k, v in _recent254[-65:]:
    if isinstance(v, (np.ndarray, torch.Tensor)):
        desc = str(tuple(v.shape))
    elif isinstance(v, dict):
        desc = "keys=" + str(list(v.keys())[:12])
    elif isinstance(v, pd.DataFrame):
        desc = str(v.shape) + " columns=" + str(list(v.columns)[:10])
    elif callable(v):
        desc = str(inspect.signature(v))
    else:
        desc = str(v)[:160]
    print(k, type(v).__name__, desc)

print("\n=== LOCAL FUNCTIONS ===")
_fns254 = [(k, v) for k, v in list(globals().items())
           if callable(v) and getattr(v, "__module__", None) == "__main__"]
for k, v in _fns254:
    try:
        print(k, inspect.signature(v))
    except Exception:
        print(k)

print("\n=== SAFE BASELINE ===")
if "safe236" in globals():
    print("safe236 keys:", list(safe236))
print("Value-head tensors:", [(n, SHAPES[n]) for n in NAMES
                               if "attr_heads.value_bin" in n])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 254\n"
            "Recovered factual26: current candidate delta +.00247 nats, within bound, "
            "but forward27 shows no useful causal-response gain. Inspecting cached update functions "
            "and earlier identification evidence to locate the lost dose signal. No weights changed. "
            "Counts: 254 Python runs, 27 forwards.\n")
```