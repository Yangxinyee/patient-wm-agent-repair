The saved kernel contains the action sensitivities, but also several rejected candidates and approximate recurrence routines. I’ll identify the exact routines used by the latest probes before testing their finite responses. If the large injection vectors saturate or reverse direction, I’ll reject them rather than treat the linear fit as a repair.

```python
needed94 = ["action_vectors4", "local_step67", "next_means_crn74",
            "action_atom_exact", "wm_set", "wm_layer", "show_impl85"]
refs94 = set()
for name94 in needed94:
    f94 = globals()[name94]
    print("\nFUNCTION", name94, "REFERENCES", f94.__code__.co_names)
    refs94.update(f94.__code__.co_names)
print("\nREFERENCED DATA")
for name94 in sorted(refs94):
    if name94 not in globals():
        continue
    v94 = globals()[name94]
    if isinstance(v94, (np.ndarray, torch.Tensor)):
        print(name94, tuple(v94.shape), str(v94.dtype))
    elif isinstance(v94, dict):
        print(name94, "dict", len(v94), "keys", str(list(v94)[:12])[:240])
    elif isinstance(v94, (list,tuple)):
        print(name94, type(v94).__name__, len(v94), str(v94[:4])[:200])
    elif isinstance(v94, (str,float,int)):
        print(name94, str(v94)[:160])
for name94 in ["local_step67", "next_means_crn74", "action_vectors4"]:
    print("\nIMPLEMENTATION", name94)
    show_impl85(name94, limit=3300)
print("\nALL ARRAY NAMES")
print([(k,tuple(v.shape)) for k,v in list(globals().items())
       if isinstance(v,(np.ndarray,torch.Tensor)) and not k.startswith("_")])
```
