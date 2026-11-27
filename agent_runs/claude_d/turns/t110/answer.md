Run 98 gives each code's median, IQR and 15 fine-bin edges, which define 16 bins. The bins are coarse: gf bins span about 9–20 mg/dL each in the middle range, and map_arterial bins span about 3–8 mmHg. `value_bin_centres` is a dict, so I'll print its keys and the centres for gf and map_arterial. Those centres convert a logit tilt into a mean shift.

```python
print(list(vbc.keys()) if isinstance(vbc,dict) else type(vbc))
for k in (list(vbc.keys()) if isinstance(vbc,dict) else []):
    v=vbc[k]; print(k, [round(x,1) for x in v] if isinstance(v,list) else v)
```