The kernel still holds the replica functions (ctx_of, e_drug_bundle, u_bundle, zmean, build_edit) and many arrays. I don't remember which array holds what, so I'll print the shapes of the likely ones: ctx, the e/u tokens at drug bundles, h, and the z_post inputs. With those I can assemble the 7 prelude tokens at drug bundles and compute the attention pattern on CPU.

```python
cand = ['CTX3','EU3','H3','H4','OUT3','OUT4','X3','X4','XIN3','XIN3d','Z3','Z4','ZO3','Zall','UI_all','UN_all','Ucomb','E_','U','TEND','P3','C3','G3','E1','E1g','Xd','Xf','Ui_only','Un_only','UB','UC','Ug','UG2','D4','TE','TS','T_','X_','Xs','W','V','P','PT']
for k in cand:
    v = globals().get(k)
    if v is None: continue
    sh = getattr(v, 'shape', None)
    print(k, type(v).__name__, sh if sh is not None else (len(v) if hasattr(v,'__len__') else ''))
```