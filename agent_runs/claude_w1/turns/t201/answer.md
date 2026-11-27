The minimum-variance decoders come out far noisier than expected. With unit response to the edits, the within-cell sd of the decoded insulin dose is 5.0 u/h at best, and 0.97 for NE. The earlier ridge decoders had errors of 0.53 and 0.18. Either the edit responses (which are nonlinear: x2 compressed for insulin, inflated for NE) are inconsistent with the natural decoding direction, or there is a bug in my setup.

Diagnostic: take the existing ridge lag-1 decoders from DEC (trained on randomly perturbed doses) and measure two things on the same rows:
- their mean response to each edit, relative to the lag-1 dose change;
- their within-cell sd.

If a ridge decoder shows response ≈ 0.86 with sd ≈ 0.7, my solver is buggy. If its response to the edits is much smaller than 0.86, the edit-regime response genuinely differs from the perturbation regime. In that case the head cannot read the edited dose cheaply.

```python
print({k_:type(v_).__name__ for k_,v_ in DEC.items()})
d0=DEC['glucose_finger']; print(type(d0).__name__, (list(d0.keys())[:6] if isinstance(d0,dict) else getattr(d0,'shape',None)))
def get_w(dobj):
    if hasattr(dobj,'coef_'): return np.asarray(dobj.coef_).ravel(), float(np.ravel(dobj.intercept_)[0]) if np.ndim(dobj.intercept_) else float(dobj.intercept_)
    if isinstance(dobj,(tuple,list)): return np.asarray(dobj[0]).ravel(), float(np.ravel(dobj[1])[0])
    return None,None
for dk,oc_code,keys in [('glucose_finger',0,keysI+keysN),('map_arterial',2,keysN+keysI)]:
    dobj=DEC[dk]; dobj=dobj[list(dobj.keys())[0]] if isinstance(dobj,dict) else dobj
    w_,b_=get_w(dobj); print(dk, 'w shape', None if w_ is None else w_.shape)
    if w_ is not None and w_.shape[0]==192:
        rows_=np.where(cf==oc_code)[0]; print(' sdw', (Rf[rows_]@w_).std().round(3), diag(w_, rows_, oc_code, keys)[2])
```