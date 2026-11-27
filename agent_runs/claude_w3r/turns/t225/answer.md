Run 212 failed because `PD` is also a DataFrame. Its 14 columns are probably the per-bundle labels for the 5,201 local bundles (stay, index, drug rates). If so, they give the targets directly, aligned with `HO`, and the 5,301-row arrays are not needed. I will print `PD`'s columns, its first rows and `recsP[0]`, and run the axis-order check inside try/except so that one failure does not hide the rest.

```python
print(type(PD).__name__, list(PD.columns)); print(PD.head(3).to_string())
print('recsP0', type(recsP[0]).__name__, str(recsP[0])[:200])
try:
    res=[]
    for pa in ['bhij','bhji']:
        o_=np.einsum(f'{pa},bjhd->bihd', patt, Vall)[:,:4]
        for hp in ['bihd','bhid']:
            H_=HO if hp=='bihd' else HO.transpose(0,2,1,3)
            res.append((pa,hp,float(np.abs(o_-H_).max()/np.abs(H_).max())))
    print(res)
except Exception as ex: print('ERR', ex)
print('patt row sums', np.round(patt[0,2].sum(1),3), np.round(patt[0,2].sum(0),3))
```