The code table gives the indices: map_cuff 5, glucose_finger 6, map_arterial 7, insulin 8, norepinephrine 9, glucose_lab 10, then the admission codes. The measurement event kind is 1 and visit is 3. The glucose fine bins don't align with the dosing thresholds, so a band step can't be read from one bin.

To find the right measurement-atom indices I need ground truth. The kernel has `E1` (100×192), `E1g` and `E_` (dicts) from earlier work. They may hold GPU-derived e_k for measurement bundles, so I'll inspect their keys and shapes.

```python
for nm in ['E1g','E_','T_','V','base']:
    v = globals().get(nm)
    if isinstance(v, dict):
        print(nm, {k: (getattr(x,'shape',None) if hasattr(x,'shape') else type(x).__name__) for k, x in list(v.items())[:8]})
    else: print(nm, type(v).__name__, getattr(v,'shape',None))
```