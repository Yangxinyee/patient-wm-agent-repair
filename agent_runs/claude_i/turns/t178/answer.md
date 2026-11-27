The signatures are readable even though the source is not. The kernel encoders are inconsistent with what I later verified. In particular, my notes record that drug START atoms use kq=1 and `categorical.action[1]`, which enc.py omits. They also record that empty sets are encoded rather than zeroed.

Rather than trust old helpers, I'll write a fresh encoder from the verified recipe in my notes and check it against the tool's z posterior means. To do that I need the structure of `atoms_by_bundle` and the names of the categorical embedding tables. This step is only inspection.

```python
k0 = list(atoms_by_bundle.keys())[:1][0]
print(type(k0), k0, atoms_by_bundle[k0][:3])
kd = [k for k in atoms_by_bundle if len(atoms_by_bundle[k])>2][:1]
print(kd, atoms_by_bundle[kd[0]] if kd else None)
print([n for n in NAMES if n.startswith('embed.')])
print('ctx', type(ctx), tuple(ctx.shape), 'X3 z err check', np.abs(X3[:5,960:965]).round(2))
```