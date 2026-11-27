The transition row matches exactly but `fixed` is off by up to 0.5, so I have the last_numeric inputs or the bins wrong. I'll solve for the 8 last_numeric inputs by least squares from the residual on a few rows and compare with has/age/zv to recover the layout.

```python
Wn_ = Wd['last_numeric.weight']; bn_ = Wd['last_numeric.bias']
for i in [5, 300, 1000, 2500, 4000]:
    r = H[i]
    res = fixed[i].double() - Wd['embed.code.weight'][r['cid']] - Wd['embed.bins.value.weight'][r['lvb']] - bn_
    f, *_ = np.linalg.lstsq(Wn_.numpy(), res.numpy(), rcond=None)
    fit = np.abs(Wn_.numpy()@f - res.numpy()).max()
    print(i, {k:r[k] for k in ['code','has','lvb','age','zv']}, 'f=', np.round(f,4), 'fit', round(fit,6), 'log1p(age)/10=', round(math.log1p(r['age'])/10,4))
```