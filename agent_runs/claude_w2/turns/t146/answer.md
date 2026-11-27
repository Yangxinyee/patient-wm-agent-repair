The test was invalid: the tool's states are not in the same row order as my bundle table. Stay ids agree on only 1% of rows and times differ by up to 27 h, so every error above compares mismatched bundles. The states are probably ordered by the request's stay list, not sorted by id. I'll reorder the states to my bundle order by matching (stay_id, minute), check the match is complete, and rerun the same variant grid. With correct alignment, one variant should give near-zero error if my core step is right.

```python
skey=pd.Series(np.arange(len(Sx)), index=pd.MultiIndex.from_arrays([S['stay_id'].astype(np.int64), np.round(S['hour']*60).astype(np.int64)]))
bkey=pd.MultiIndex.from_arrays([B_T.stay_id.values.astype(np.int64), np.round(B_T.t.values*60).astype(np.int64)])
print('dup S keys', skey.index.duplicated().sum(), 'missing', (~bkey.isin(skey.index)).sum())
perm_S=skey.reindex(bkey).values
SxA=Sx[perm_S]
print('aligned hour diff', np.abs(S['hour'][perm_S]-B_T.t.values).max(), (S['stay_id'][perm_S]==B_T.stay_id.values).mean())
print('z check (statez vs my zpost):')
res_c=[]
for ic in [False,True]:
  for au_ in [1.0,60.0,3600.0]:
    for am in ['kadd','inadd','kin']:
      for uz in [False,True]:
        eo,eh,_,_,_=core_eval(B_T,E_T,U_T,hasu_T,SxA,W,include_cur=ic,age_unit=au_,age_mode=am,uzero=uz)
        res_c.append((round(eo,5),round(eh,5),ic,au_,am,uz))
res_c.sort(); print(res_c[:8]); print(res_c[-2:])
```