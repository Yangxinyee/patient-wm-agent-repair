The kernel has `enc_eu2`, `zcheck2` and `gf` from my last session. ACTMAP shows that gelu is the activation in gap, both encoders and z_post, which is why `e` matched exactly. I'll read the source of the latest encoder and z-check functions so I can test the remaining z_post hypotheses directly. The candidates are input order, the gap of the first bundle, and whether ctx uses `out` from before or after the step. I'll truncate the printout.

```python
import inspect
for f in ['enc_eu2','zcheck2','gf','fix_atom2']:
    try:
        src_ = inspect.getsource(globals()[f]); print('#### ',f, len(src_)); print(src_[:1400])
    except Exception as ex:
        print(f, 'no source', ex)
```