The current diagnostic tables survived the restart, so there is no need to rebuild them from scratch. The function listing was obscured by large default arguments. I’ll inspect compact summaries of the head and centering fits next; this should reveal whether the centering term reads dose-sensitive state or only pre-treatment history.

```python
print("=== FUNCTIONS, WITHOUT DEFAULT VALUES ===")
for _n216, _v216 in list(globals().items()):
    if callable(_v216) and getattr(_v216, "__module__", None) == "__main__":
        try:
            _params216 = list(inspect.signature(_v216).parameters)
        except Exception:
            _params216 = []
        print(_n216, "(" + ", ".join(_params216) + ")")

def summarize216(obj, prefix="", depth=0):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if depth < 1 and isinstance(value, dict):
                print(prefix + str(key) + ":")
                summarize216(value, prefix + "  ", depth + 1)
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                print(prefix + str(key), "shape", tuple(value.shape))
            elif isinstance(value, (int, float, str, bool, type(None))):
                print(prefix + str(key), repr(value))
            elif isinstance(value, (list, tuple)) and len(value) <= 12:
                print(prefix + str(key), repr(value)[:250])
            else:
                print(prefix + str(key), type(value).__name__)

for _n216 in ["head_audit", "history_audit", "encoder_audit", "sensor_neurons"]:
    print("===", _n216, "===")
    summarize216(globals()[_n216])

print("=== HEAD ROWS ===")
print(head_rows.head(3).to_string(index=False))
print("=== PRIVATE AND RESERVED COORDINATES ===")
print("priv", priv, "pairs", private_pairs, "reserved", reserved)
print("=== HEAD INPUT ARRAY RANGES ===")
for _n216 in ["head_ln", "head_zp", "head_prev"]:
    _a216 = np.asarray(globals()[_n216])
    print(_n216, "shape", _a216.shape,
          "range", float(np.nanmin(_a216)), float(np.nanmax(_a216)))

print("=== IDENTIFICATION CHECKS FROM NOTEBOOK ===")
for _s216 in Path("/work/notes.md").read_text().split("\n## "):
    if _s216.startswith("Runs 17") or _s216.startswith("Sensor checks"):
        print(("\n## " + _s216)[:2500])
```
