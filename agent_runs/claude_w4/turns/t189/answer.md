`step_full` with default flags misses yt by 19% (relative), so the defaults are not the best-fitting variant. My notes say the version with the age embedding on keys and values fitted with R2 0.998. I'll compare a few flag settings of `step_full`, `step_var` and `step_parts` on the same 60 rows to find the accurate one.

```python
def relerr(res, rw):
    y_, o_ = res[0], res[1]
    return float((y_-yt[rw]).norm()/yt[rw].norm()), float((o_-ot[rw]).norm()/ot[rw].norm())
tests = {
 'full pre=T': lambda: step_full(rw, E0[rw], htf[rw], pre=True),
 'full incl=F': lambda: step_full(rw, E0[rw], htf[rw], incl=False),
 'var default': lambda: step_var(rw, E0[rw], htf[rw]),
 'var bpre=F': lambda: step_var(rw, E0[rw], htf[rw], block_pre=False),
 'var cpre=F': lambda: step_var(rw, E0[rw], htf[rw], coda_pre=False),
 'var pre=F': lambda: step_var(rw, E0[rw], htf[rw], pre=False),
 'parts default': lambda: step_parts(rw, E0[rw], htf[rw]),
}
for k,fn_ in tests.items():
    try:
        r_ = fn_(); print(k, ['%.4f'%x for x in relerr(r_, rw)])
    except Exception as ex: print(k, 'ERR', str(ex)[:100])
```