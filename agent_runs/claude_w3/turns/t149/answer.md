`zpost_err` takes an e mode ('own', 'next' or 'zero'). The 'next' mode suggests I had suspected the saved z might pair with the following bundle's encoding. I'll call `zpost_err` with each mode. The mode with a near-zero error tells me the alignment convention between saved states and bundles; if all are large, the old helper is also unvalidated and I'll test that alignment shift in my own code.

```python
for em in ['own','next','zero']:
    try:
        r_ = zpost_err(em); print(em, r_ if len(str(r_)) < 300 else str(r_)[:300])
    except Exception as ex: print(em, 'ERR', str(ex)[:150])
```