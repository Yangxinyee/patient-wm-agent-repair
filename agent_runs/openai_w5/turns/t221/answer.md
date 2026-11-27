The recovered objects include the current replay function and finite-dose audits, while the record-based targets are roughly −11.9/+5.9 for insulin and +4.5/−2.2 for norepinephrine on tuning A. I’ll inspect those audits and the replay’s inputs next. Linear private-channel transport would shift attention toward the head’s exposure-centering model rather than drug encoding.

```python
print("=== FINITE-DOSE AUDITS ===")
import pandas as _r195_pd
print(_r195_pd.DataFrame(audit189).round(7).to_string(index=False))
print("Refreshed-cache audit191:")
print(audit191)

print("=== REPLAY IMPLEMENTATION ===")
compact_restart171(wm_step_boundary176, maxchars=4900)

print("=== EARLIER HELPER FUNCTIONS ===")
for _r195_n, _r195_o in _r194_functions[:31]:
    print(_r195_n, str(_r194_inspect.signature(_r195_o)),
          "globals:", _r195_o.__code__.co_names[:14])

print("=== SENSOR SPECS ===")
_r195_items = list(globals().items())
_r195_spec_objects = []
for _r195_n, _r195_o in _r195_items:
    if isinstance(_r195_o, (list, tuple)) and len(_r195_o) == 4:
        if all(isinstance(_r195_s, dict) and "row_signal" in _r195_s
               and "proxy" in _r195_s for _r195_s in _r195_o):
            _r195_spec_objects.append((_r195_n, _r195_o))
for _r195_n, _r195_specs in _r195_spec_objects:
    print("Specs:", _r195_n)
    for _r195_s in _r195_specs:
        print({
            _r195_k: (
                tuple(_r195_v.shape) if hasattr(_r195_v, "shape")
                else _r195_v
            )
            for _r195_k, _r195_v in _r195_s.items()
            if _r195_k != "tilt"
        })

print("=== REPLAY CACHE DIMENSIONS ===")
for _r195_n in ("wm_e4", "wm_u4", "wm_z4", "wm_df4", "wm_W",
                "wm_weights", "wm_indices4", "wm_start4", "wm_prev4"):
    if _r195_n in globals():
        _r195_o = globals()[_r195_n]
        print(_r195_n,
              tuple(_r195_o.shape) if hasattr(_r195_o, "shape")
              else list(_r195_o)[:12] if isinstance(_r195_o, dict)
              else repr(_r195_o)[:200])
```