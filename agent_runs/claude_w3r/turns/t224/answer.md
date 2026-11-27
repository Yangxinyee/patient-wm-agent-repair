Two problems. The 5,201-bundle arrays (`HO`, `patt`, `Vall`) are not aligned with the 5,301-bundle targets in `T_comb`; the difference of 100 is probably the first bundle of each of the 100 stays. And my recomputation does not match `HO` (relative error 1.19), so I have the axis order wrong. Next I will check what `recsP` stores for each of the 5,201 bundles, so I can align them. I will also try the axis orders of `patt` and `HO` against a direct recomputation from `Vall`.

```python
print('recsP0', recsP[0], recsP[1], '| len T', len(T_comb['ins_s']), '| PD0', np.round(PD[0],3))
cands_=[]
for pa in ['bhij','bhji']:
    for ho in ['bihd','bhid']:
        o_=np.einsum(f'{pa},bjhd->{ho}',patt[:,:,:4,:] if pa=='bhij' else patt[:,:,:,:4],Vall)
        o2_=o_ if ho=='bihd' else o_.transpose(0,2,1,3)
        for hperm in ['bihd','bhid']:
            H_=HO if hperm=='bihd' else HO.transpose(0,2,1,3)
            cands_.append((pa,ho,hperm,float(np.abs(o2_-H_).max()/np.abs(H_).max())))
print(cands_)
print('patt sums q-axis', patt[0,0].sum(1)[:3], 'k-axis', patt[0,0].sum(0)[:3])
```