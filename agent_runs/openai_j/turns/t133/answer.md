The provisional fixed-dose targets are larger than the original rollouts, especially for norepinephrine. Insulin’s longer-term target remains less secure. I’ll inspect the corrected fitting inputs next; a useful candidate must strengthen dose sensitivity while leaving factual predictions nearly unchanged.

```python
import inspect

audit_arrays = []
audit_funcs = []
for audit_name, audit_obj in list(globals().items()):
    if audit_name.startswith("_") or audit_name.startswith("audit_"):
        continue
    if isinstance(audit_obj, (pd.DataFrame, pd.Series, np.ndarray, dict)):
        audit_arrays.append((audit_name, audit_obj))
    elif callable(audit_obj) and getattr(audit_obj, "__module__", "") == "__main__":
        audit_funcs.append((audit_name, audit_obj))

print("=== MOST RECENT CACHES ===")
for audit_name, audit_obj in audit_arrays[-75:]:
    if isinstance(audit_obj, dict):
        print(audit_name, "dict keys:", list(audit_obj)[:12])
    else:
        print(audit_name, type(audit_obj).__name__, audit_obj.shape)
print("\n=== RECENT HELPER SIGNATURES ===")
for audit_name, audit_obj in audit_funcs[-25:]:
    print(audit_name, inspect.signature(audit_obj))

print("\n=== HEAD AND CONTRAST HELPERS ===")
for audit_name in ["reconstruct_reading_head", "repair_local_step",
                   "repair_reading_features", "repair_predict", "fit_head_contrast"]:
    audit_obj = globals().get(audit_name)
    if callable(audit_obj):
        print(audit_name, inspect.signature(audit_obj),
              "referenced names:", audit_obj.__code__.co_names,
              "arguments/locals:", audit_obj.__code__.co_varnames)
        try:
            print(inspect.getsource(audit_obj)[:2200])
        except (OSError, TypeError):
            print("Source unavailable; bytecode retained.")

print("\n=== ORIGINAL ROLLOUT TABLE COLUMNS ===")
for audit_k in [1, 2, 7, 9, 11]:
    audit_p = Path(f"/work/fwd_{audit_k}/dose_edit_original.csv")
    if audit_p.exists():
        audit_df = pd.read_csv(audit_p)
        print(audit_k, audit_df.shape, audit_df.head(2).to_dict("records"))
print("\nActive edits:", edited())
```
