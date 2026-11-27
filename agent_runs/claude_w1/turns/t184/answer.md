gen_samples returns, per finger/arterial reading, (code, stay, hour, prev, cur, dose lags 1–3, with_code[192]); G3 holds 1,500 train stays. For gating and for compensation per (code, last bin), I also need lab and cuff rows. The kernel has `wc_vec` and `gen_rows_all`, which look like they produce rows for all four codes. I'll check their signatures and run gen_rows_all on one stay. A row count near 60 with all four codes (as in run 167) would confirm that it covers all readings.

```python
for fn in ['wc_vec','gen_rows_all','lastbin_of','cid']:
    f_=globals().get(fn)
    if f_ is None: print(fn,'MISSING'); continue
    if callable(f_):
        co=f_.__code__; print(fn, inspect.signature(f_), '| vars', co.co_varnames[:18])
    else: print(fn, type(f_).__name__)
print(type(allrows).__name__, len(allrows) if hasattr(allrows,'__len__') else '')
try:
    ra = gen_rows_all(G3[s0]) 
    print(type(ra).__name__, len(ra)); r0=ra[0]; print(type(r0).__name__, len(r0), [(type(x).__name__, getattr(x,'shape',None)) for x in r0][:10])
    from collections import Counter; print(Counter([r[0] for r in ra]))
except Exception as ex: print('ERR', repr(ex)[:300])
```