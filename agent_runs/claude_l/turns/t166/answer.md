The run failed before anything else ran: `value_bin_centres` in the vocab is a dict, probably one list of centres per code, not one shared list. I'll print its structure, then convert it per code and rerun the head-replica grid.

```python
vc = VOCAB['value_bin_centres']; print(type(vc), list(vc.keys())[:10])
k0 = list(vc.keys())[0]; print(k0, type(vc[k0]), (vc[k0] if len(str(vc[k0]))<600 else str(vc[k0])[:600]))
```