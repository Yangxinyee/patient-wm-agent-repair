The z_post module is two layers, Linear(576→192) → activation → Linear(192→64), with the 64 outputs taken as 32 means and 32 log-sds. The kernel still holds state arrays, `X` with 4932 bundles × 992 numbers, and per-bundle stay and hour arrays. I'll read the rest of my setup code and check the units of the time-flow parameters. A 0.28 error with the encoders exact points to the context term (dt units, the gap input, or the decay) rather than the encoders. If `exp(-core.flow.a)` gives timescales from 600 s to 6e7 s, dt is in seconds. If it gives 0.17 h to 17,520 h, dt is in hours, and my seconds-based context would explain the error.

```python
print(s[1500:])
for d in ['fwd_1','fwd_2','fwd_3','fwd_4']:
    print(d, os.listdir('/work/'+d))
fa = load('core.flow.a'); print('flow.a', fa.shape, float(fa.min()), float(fa.max()))
print('timescale range (exp(-a)):', float(np.exp(-fa.max())), float(np.exp(-fa.min())))
print(json.load(open('/work/batches.json')).keys() if isinstance(json.load(open('/work/batches.json')),dict) else len(json.load(open('/work/batches.json'))))
print(sid_arr[:5], hr_arr[:12])
```