The tensor list gives me workable handles:

- `z_in.weight` (192×32) maps the 32-dimensional z posterior mean into the update. Its input space is directly observable through the states tool.
- `z_post.2.weight` (64×192) produces the posterior mean.
- `z_post.0.weight` (192×576) reads [ctx, e_k, u_k].
- `encode_action.*` is specific to drug atoms, but its output u_k is not observable.

My two current operations scale only the z_dose and z_rate columns of the numeric embedding. They help little because the bins and the action encoder's normalisation dilute them.

My working hypothesis is that the z posterior at drug bundles carries the surprising part of the dose (the dose innovation). If I find a direction in z that tracks the dose innovation and barely varies at other bundles, scaling `z_in` along it would strengthen a dose response that does not double count. Next I will check how dose z-scores are defined and what the saved states file and panels contain. If the states show a z direction that tracks dose innovation, the hypothesis holds; if z mostly tracks the raw dose or tier, it fails.

```python
print(VOCAB['numeric_note'])
print(json.dumps(VOCAB['numeric'])[:1500])
print(VOCAB['value_bin_note'])
z3 = np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
for k in z3.files:
    a = z3[k]; print(k, a.shape, a.dtype, a[:3] if a.ndim==1 else '')
print(json.load(open('/work/fwd_3/summary.json')))
for nm_ in ['Ptu','P','panel','S','Z','st','dz','proj','bD_z','Bz']:
    if nm_ in globals():
        o = globals()[nm_]
        print(nm_, type(o), getattr(o,'shape',None), list(o.columns)[:40] if hasattr(o,'columns') else '')
```