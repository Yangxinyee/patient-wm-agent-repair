The rollout gain is real, but the factual constraint still fails. The last inspection was truncated before reaching the fitted coefficients. I’ll retrieve only those arrays and helpers, then test whether centering the correction on record-based dose innovations reduces factual error without losing dose sensitivity.

```python
import inspect
import re

print("ARRAY INVENTORY")
recovery_arrays = []
for recovery_name, recovery_obj in list(globals().items()):
    if recovery_name.startswith("_") or recovery_name.startswith("recovery"):
        continue
    if isinstance(recovery_obj, (np.ndarray, torch.Tensor)):
        recovery_arrays.append(f"{recovery_name}:{tuple(recovery_obj.shape)}")
print("\n".join(recovery_arrays))

print("\nRECENT FIT OBJECTS")
for recovery_name, recovery_obj in list(globals().items()):
    if re.search(r"(46|47|contrast|anchor|direction|strength|beta|rho|phi)", recovery_name.lower()):
        if isinstance(recovery_obj, (int, float, str)):
            print(recovery_name, str(recovery_obj)[:200])
        elif isinstance(recovery_obj, dict):
            print(recovery_name, "keys:", list(recovery_obj)[:20])
        elif hasattr(recovery_obj, "coef_"):
            print(recovery_name, type(recovery_obj).__name__, "coef shape", recovery_obj.coef_.shape)

print("\nHELPER SOURCES")
for recovery_function in [prelude46, edited_action46, next_means45, coda46, schedule_readout]:
    try:
        recovery_source = inspect.getsource(recovery_function)
        print(recovery_function.__name__, recovery_source[:2400])
    except Exception as recovery_error:
        print(recovery_function.__name__, "source unavailable:", type(recovery_error).__name__)

print("\nFACTUAL COLUMNS", factual4_original.columns.tolist())
print(factual4_original.head(1).to_string(index=False))

with open("/work/notes.md", "a") as recovery_log:
    recovery_log.write(
        "\n## Run 50: targeted coefficient recovery\n"
        "Prior recovery output was truncated. Restricted inspection to retained "
        "arrays and trial helpers before testing factual-innovation centering. "
        "No new edits. Python runs: 50; forwards: 5.\n"
    )
```