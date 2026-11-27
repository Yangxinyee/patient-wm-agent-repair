The u vectors at drug bundles lie almost exactly on a low-dimensional affine subspace:
- With 10 principal components the leftover variance is 1.2e-5 of the total.
- With 25 components it is 7.7e-7, and the largest leftover norm is 0.017, against a typical |u| of 11.5. The real drug-bundle u's fit equally well (rms 0.0066).

So z_post plus z_in can make the z token a near-exact copy of u_k at drug bundles, using about 25 of z's 32 dimensions:
- 1 dimension for "a drug is present";
- 25 for the principal-component coefficients;
- 2 for the insulin and NE surprises.

This confirms the z-clone mechanism is feasible.

Next I need to reuse my exact encode_action copy to compute u for any (insulin rate, NE rate) pair, including the edited rates ×0 to ×3. I will then build the basis and the gates on those.

The grid's rate ranges printed as NaN, which suggests absent drugs are stored as NaN. So in this run I will:
- list the signatures of the u-building helpers, printing parameter names only, so I don't flood the output again;
- print the grid's ranges, ignoring NaN.

```python
import inspect, numpy as np
for nm_f in ['Up', 'Uc', 'U_of', 'u_set', 'u_from_atoms', 'ins_atom', 'ne_atom', 'readout', 'fIu', 'fNu', 'u_bundle', 'drug_atom_parts', 'gate_fit', 'build_edit', 'zmean']:
    fn = globals().get(nm_f)
    if fn is None:
        print(nm_f, 'missing'); continue
    try:
        ps = list(inspect.signature(fn).parameters.items())
        print(nm_f, [(k, (repr(p.default)[:20] if p.default is not inspect._empty else '')) for k, p in ps])
    except Exception as ex:
        print(nm_f, 'sig error', ex)
aG = np.asarray(aG2, float); bG = np.asarray(bG2, float)
print('grid ins rate: nan', int(np.isnan(aG).sum()), 'min', np.nanmin(aG), 'max', np.nanmax(aG), 'distinct', len(np.unique(aG[~np.isnan(aG)])))
print('grid NE rate: nan', int(np.isnan(bG).sum()), 'min', np.nanmin(bG), 'max', np.nanmax(bG), 'distinct', len(np.unique(bG[~np.isnan(bG)])))
print('grid_pairs[:3]', grid_pairs[:3], ' rI0', np.asarray(rI0)[:5], len(rI0), ' rN0', np.asarray(rN0)[:5], len(rN0))
print('RI_d stats', np.nanmin(RI_d), np.nanmax(RI_d), ' RN_d', np.nanmin(RN_d), np.nanmax(RN_d), ' thI', thI, 'thN', thN)
```