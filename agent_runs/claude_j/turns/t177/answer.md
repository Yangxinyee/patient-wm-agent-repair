The earlier states (`Xt`, 5116 bundles × 992) and the bundle lists (`BUN`) are still in the kernel. `CFG` holds the (unit, rate_unit) indices per drug.

Next I need the structure of one bundle in `BUN`, so I can run a full recursion that computes z myself and compare it with the tool's states. I will print the type and keys of the first few bundles of the first stay, briefly.

```python
s0=list(BUN.keys())[0]; b=BUN[s0]
print(s0, type(b).__name__, len(b))
x=b[0] if isinstance(b,list) else b
print(type(x).__name__, (list(x.keys()) if isinstance(x,dict) else [type(t).__name__ for t in x])[:12])
for t in (b[:3] if isinstance(b,list) else []):
    print({k:(tuple(v.shape) if hasattr(v,'shape') else v) for k,v in (t.items() if isinstance(t,dict) else enumerate(t))})
```