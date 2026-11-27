None of the memory rules reproduce the tool exactly. `'e'` is closest (median 0.82), while earlier sessions reached about 1e-6. The likely cause: several global configuration flags (such as `agemode`, `memnorm`, `utok`, `mode`) steer the replica, and a later experiment left them at non-matching values. I'll list which flags each replica function reads and their current values. Then I can test the alternatives systematically.

```python
fl=set()
for fname in ['tlayer','block','mha','ffn','encode','bundle_vecs','drug_atom','drug_end','visit_atom','meas_atom','linp','lnp','mlp','lin','step']:
    f=globals()[fname]
    for n in f.__code__.co_names:
        if n in globals() and isinstance(globals()[n],(bool,str,int,float)): fl.add((fname,n))
print(sorted(fl))
print({n:globals()[n] for n in sorted({x[1] for x in fl})})
```