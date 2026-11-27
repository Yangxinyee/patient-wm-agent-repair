The architecture confirms that doses affect the recurrent state through a separate action encoder, while reading predictions also use a last-reading shortcut. That gives us a possible targeted repair path, but does not establish the correct effects. I’ll recover the retained code and data objects next. The immediate test is whether the remaining posterior errors can be isolated to a few event conventions.

```python
import inspect, re
lines = []
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if callable(v) and getattr(v, "__module__", None) == "__main__":
        try:
            lines.append(f"FN {k}{inspect.signature(v)}")
        except Exception:
            lines.append(f"FN {k}")
    elif isinstance(v, pd.DataFrame):
        lines.append(f"DF {k} {v.shape} columns={list(v.columns)[:12]}")
    elif isinstance(v, dict) and len(v) < 100:
        lines.append(f"DICT {k} keys={list(v)[:14]}")
    elif (isinstance(v, np.ndarray) or torch.is_tensor(v)) and re.search(
            r"ctx|post|bundle|count|pre_|recipe|state|lag|kernel|panel|innovation|resid|feature|head|dose", k):
        lines.append(f"ARR {k} {tuple(v.shape)}")
inventory61 = "\n".join(lines)
print(inventory61[:8800])
print("inventory characters:", len(inventory61))
print("raw record columns:", pd.read_csv("/data/train.csv", nrows=1).columns.tolist())
print("current edited tensors:", edited())
```
