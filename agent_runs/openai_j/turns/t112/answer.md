The tensor shapes reveal eight inputs to `last_numeric`, although the architecture describes only four fields. That unresolved layout could explain the biased reconstruction. I will inspect the current helper and layout tests before fitting anything else. A credible layout should reduce bias across all four sensors, not merely improve pooled NLL.

```python
import inspect
import dis
import io
import numpy as np

function_names = [
    name for name, value in list(globals().items())
    if callable(value) and hasattr(value, "__code__")
]
print("=== FUNCTION NAMES ===")
print(function_names)

print("\n=== BEST RECONSTRUCTION TESTS ===")
for name, criterion in [
    ("layout_table93", "scaled_rmse"),
    ("flag_table94", "rmse"),
    ("integration_table", "scaled_mean_rmse"),
    ("core_check_table", "out_rmse")
]:
    if name in globals():
        table = globals()[name]
        print(name)
        print(table.sort_values(criterion).head(5).to_string(index=False))

print("\n=== EIGHT-COLUMN CACHED ARRAYS ===")
for name, value in list(globals().items()):
    if hasattr(value, "shape") and len(value.shape) == 2 and value.shape[1] == 8:
        print(name, tuple(value.shape))
        try:
            print(value[:2])
        except Exception:
            pass

print("\n=== HEAD RECONSTRUCTION IMPLEMENTATION ===")
fn = reconstruct_reading_head
try:
    text = inspect.getsource(fn)
except (OSError, TypeError):
    stream = io.StringIO()
    dis.dis(fn, file=stream)
    text = stream.getvalue()
print(text[:5000])

print("\n=== HEAD-RELATED FUNCTION SIGNATURES ===")
for name in function_names:
    if any(s in name for s in ["reading", "head", "repair_changed"]):
        fn = globals()[name]
        print(name, inspect.signature(fn))
```