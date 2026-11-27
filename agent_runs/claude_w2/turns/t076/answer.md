The greedy search finds no sensible register layout. It drifts to scattered squared terms for a small gain (NLL 1.4526). The register is probably not the main gap.

The shortfall is specific to MAP: my head matches the tool for finger glucose but is less sharp for both MAP codes. So something code-specific may be wrong. The best candidate is the type embedding. I fitted type 1 jointly across all codes, and under SiLU, but each code, or MAP versus glucose, might use its own type index.

Next I search all 15 type indices separately for each code under GELU, using the age-in-minutes register. A type index that pulls MAP's NLL down to the tool's value would settle it.

```python
f_m = np.stack([feats2[a_] for a_ in ['has','la_m','zl','has','zero','zero','zero','zero']],1)
best_ty = {}
for ty in range(15):
    pmv, _, nlv, _, _, _ = run13(f_m, lbL, obL, ty=ty)
    d_ = fa4.assign(n_=nlv, ad=np.abs(pmv-tgt)).groupby('event')[['n_','ad']].mean()
    for ev in d_.index:
        if ev not in best_ty or d_.loc[ev,'n_'] < best_ty[ev][0]: best_ty[ev] = (round(d_.loc[ev,'n_'],4), round(d_.loc[ev,'ad'],2), ty)
print(best_ty); print(fa4.groupby('event').nll.mean().round(4).to_dict())
```