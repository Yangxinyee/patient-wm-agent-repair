The error is inside `encode`. One of the global helper names it calls was overwritten by an integer in my recent runs, because I reused short variable names. I'll list the globals `encode`, `bundle_vecs` and `step` reference, and flag any that are no longer callable. Then I'll know which helper to restore, if I can.

```python
for f in [encode, bundle_vecs, step, meas_atom, drug_atom, mlp, ln, lin, mha, ffn, tlayer, block]:
    bad_=[n for n in f.__code__.co_names if n in globals() and not callable(globals()[n]) and not hasattr(globals()[n],'__getitem__') and not hasattr(globals()[n],'__dict__')]
    ints_=[f"{n}:{type(globals()[n]).__name__}" for n in f.__code__.co_names if n in globals() and isinstance(globals()[n],(int,float,str))]
    print(f.__name__, 'non-callable scalars:', ints_[:8])
```