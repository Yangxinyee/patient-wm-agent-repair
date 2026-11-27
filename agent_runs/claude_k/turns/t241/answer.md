Run 224 isolates the rebuild error to the atoms. Bundles with a single atom rebuild almost exactly (error 0.002). Bundles with 2–4 atoms have errors around 0.4.

So the action and empty observation encodings are right for one START atom. Bundles with more atoms hold something my rebuild leaves out, most likely the END atoms of the infusions that stop at that bundle. These could be encoded either in the action set or in the observation set.

Next I'll print what `u_rec` builds, from its constants. I'll also print the raw records of two 4-atom bundles: the drug rows starting at that time, and the drug rows whose infusion ends then. That tells me which variants to try for placing the END atoms.

```python
print('u_rec consts',u_rec.__code__.co_consts[:20])
print('datom args',datom.__code__.co_varnames[:datom.__code__.co_argcount],'consts',[c for c in datom.__code__.co_consts if isinstance(c,(str,int,float))][:20])
trE=pd.read_csv('/data/train.csv',usecols=['stay_id','hour','event','rate','dose','end_hour'])
trE=trE[trE.stay_id.isin(set(np.unique(stA)))&trE.event.isin(['insulin','norepinephrine'])]
j4=np.where(nat==4)[0][:2]
for j in j4:
    s_,h_=stA[dr[sub_[j]]],hrA[dr[sub_[j]]]
    st_rows=trE[(trE.stay_id==s_)&(np.abs(trE.hour-h_)<1e-3)]; en_rows=trE[(trE.stay_id==s_)&(np.abs(trE.end_hour-h_)<1e-3)]
    print('bundle',s_,round(h_,3),'START:',st_rows[['event','rate','end_hour']].values.tolist(),'END:',en_rows[['event','rate','hour']].values.tolist())
```