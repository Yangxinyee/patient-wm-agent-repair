Using the empty-set encoding for bundles without a drug start, u now matches in 99.7% of bundles. About 15 bundles still miss by up to 2.04. I'll list their compositions and times, to see whether the cause is something special such as duplicate atoms, stay boundaries or rates at the cap.

```python
bad=np.where(errU2>1e-4)[0]
print(len(bad))
for i in bad[:15]:
    r=recs[i]; atoms=BUN[r[0]][r[1]][1]; t_=BUN[r[0]][r[1]][0]
    print(i, r[0], r[1], t_, [(a['code'],a['kind'],a['rb'],round(a['num'][4],2)) for a in atoms if a['act']], round(errU2[i],3))
```