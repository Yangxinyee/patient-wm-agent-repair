No flag setting of `step_full` gets below about 17% error. My notes record the best variant as the one with the age embedding on keys and values ('ageKV' / 'lnM_ageKV'), and `step_cv(rws, E, htf, mode)` probably selects it by name. I'll try those mode names on the same rows.

```python
for mode in ['ageKV','lnM_ageKV','lnM','noage','full','base']:
    try:
        r_ = step_cv(rw, E0[rw], htf[rw], mode)
        if isinstance(r_, (tuple,list)) and len(r_)>=2 and hasattr(r_[0],'shape') and r_[0].shape==yt[rw].shape:
            print(mode, ['%.4f'%x for x in relerr(r_, rw)])
        else:
            print(mode, type(r_).__name__, [getattr(x,'shape',type(x).__name__) for x in (r_ if isinstance(r_,(tuple,list)) else [r_])][:4])
    except Exception as ex: print(mode, 'ERR', str(ex)[:120])
```