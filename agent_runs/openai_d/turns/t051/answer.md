The compact inventory succeeded. We have the original states, reading-head features, and dose-innovation caches needed for a controlled trial. I’ll inspect their alignment and the actual posterior tensor names next. Correct alignment would support an earlier intervention; an indexing mismatch would invalidate the decoder results.

```python
print("=== CORE TENSOR NAMES ===")
for rec_name in NAMES:
    if rec_name.startswith("core."):
        print(rec_name, SHAPES[rec_name])

print("\n=== IMPORTANT CACHE STRUCTURES ===")
for rec_name in [
    "states0", "dml_cache", "wr_instruments", "repair_aligned",
    "repair_decoder_fits", "repair_state_blocks", "rh_chosen", "best_reconstruction"
]:
    rec_obj = globals().get(rec_name)
    print(rec_name)
    if isinstance(rec_obj, dict):
        for rec_key, rec_val in rec_obj.items():
            if isinstance(rec_val, dict):
                print(" ", rec_key, "dict keys", list(rec_val))
            elif hasattr(rec_val, "shape"):
                print(" ", rec_key, type(rec_val).__name__, tuple(rec_val.shape))
            elif isinstance(rec_val, (list, tuple)):
                print(" ", rec_key, type(rec_val).__name__, "length", len(rec_val))
            else:
                print(" ", rec_key, str(rec_val)[:180])

print("\n=== FUNCTION DETAILS WITHOUT DEFAULT VALUES ===")
for rec_name in ["fn", "rc_lin", "rc_norm", "rh_register", "rh_eval", "wnorm"]:
    rec_obj = globals().get(rec_name)
    if inspect.isfunction(rec_obj):
        print(rec_name, "names:", rec_obj.__code__.co_names,
              "constants:", [c for c in rec_obj.__code__.co_consts
                             if isinstance(c, (str, int, float, tuple))])
        try:
            rec_src = inspect.getsource(rec_obj)
            print(rec_src[:1600])
        except Exception:
            pass

print("\n=== SAMPLE IDENTIFIERS AND ALIGNMENT ===")
print("first train IDs:", train_ids[:5].tolist())
print("next training batch:", train_ids[100:200].tolist())
print(head_meta.head(3).to_string(index=False))
print("query/with-code norms:",
      np.quantile(np.linalg.norm(_r40_q, axis=1), [0, .5, 1]),
      np.quantile(np.linalg.norm(repair_wc_mean, axis=1), [0, .5, 1]))
```