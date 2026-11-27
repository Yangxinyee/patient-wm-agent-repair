The action encoder has no linear path: u = norm(mlp2(Σ mlp1(atom)/√n + size(log(1+n)))), with GELU layers in both MLPs. A dead hidden unit in mlp1 or mlp2 can still act as a gated linear channel, GELU(G+P) − GELU(G) ≈ P. Since u is zero at bundles with no action, nothing would leak at no-action bundles.

This works only if the atom vector holds a feature linear in the rate. A log-scaled or binned rate would not give doses that scale linearly. Next I will read how atoms are embedded (the text just before "Set encoders") and the rate normalisation in VOCAB.

```python
j0=arch.find('**Set encoders**'); print(arch[max(0,j0-2600):j0])
print({k:(str(v)[:300]) for k,v in VOCAB.items() if k in ('normalisation','normalization','sizes','norm')})
print(list(VOCAB.keys()))
```