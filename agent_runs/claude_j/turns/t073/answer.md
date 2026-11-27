The replica's z-posterior matches the GPU to about 1e-7 on bundles that hold only measurements. Bundles with drug atoms, especially those with infusion END plus START records, are off by about 5e-3, and up to 0.1. So the measurement encoder is right and the small mismatch is in how I encode drug atoms (u_k), most likely the END atoms. Next I'll group the errors by bundle composition and print my drug-atom encoding code to find the difference. If the errors are confined to END-containing bundles, the END atom's attributes are the thing to fix.

```python
import inspect
print(E_.groupby('what')['err'].agg(['count','median','max']).sort_values('count',ascending=False).head(15))
for fnm in ['drug_atom2','drug_end','bundle_vecs']:
    try:
        print('-----',fnm); print(inspect.getsource(globals()[fnm]))
    except Exception as ex_:
        print(fnm, 'no source', ex_)
```