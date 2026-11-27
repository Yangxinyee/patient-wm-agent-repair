`zcheck2(s, gapf, flow_unit, use_flow)` rebuilds ctx from the previous bundle's state in `Xs`, encodes the bundle with `enc_eu2`, and compares the z_post output with the recorded z. I'll check the shapes of `BUN` and `row_start` and run it once with the documented gap input `log(dt)/10`. That gives me a baseline error before I test changes to the input layout.

```python
print(type(BUN), len(BUN), type(BUN[0]), len(BUN[0]), type(BUN[0][0]))
print(BUN[0][0][0] if isinstance(BUN[0][0],(tuple,list)) else BUN[0][0])
rs = globals().get('row_start'); print(type(rs), (rs[:5] if hasattr(rs,'__getitem__') else rs))
r = zcheck2(0, lambda dt: math.log(dt)/10)
print(type(r)); 
try: print(np.round(np.array(r)[:12],3), np.median(np.array(r)))
except Exception as ex: print(ex, r)
```