The local rollout runs: 13.8 s for 30 stays × 5 arms × 20 paths. On these 30 stays it gives insulin ×2 −2.90 and ×½ +1.48, the same order as the tool's 100-stay −2.14/+1.26. The comparison then crashed because `shift` clashes with a pandas method name. I'll redo the comparison with column indexing, on the same 30 stays and rows. I'll check that reached flags agree, compare the tool's mean shift with mine, and compare factual predicted means. Agreement within path noise would validate the engine.

```python
for drug,outc,aa,edn in [('insulin','glucose_finger',[1,2],['insulin_x2','insulin_half']),('norepinephrine','map_arterial',[3,4],['ne_x2','ne_half'])]:
    rows,reached,o=readout(R30,res30,arms5,drug,outc)
    A_=R30['A']; st_=R30['stays'][A_.si.values[rows]]; hr_=np.round(A_.t.values[rows]).astype(int)
    for a,en in zip(aa,edn):
        tool=de1[de1.stay_id.isin(ids30)&(de1.edit==en)]
        mine=pd.DataFrame({'stay_id':st_,'hour':hr_,'mine_fact':np.nanmean(o[0],1),'mine_sh':np.nanmean(o[a]-o[0],1),'mine_reached':reached})
        mg=tool.merge(mine,on=['stay_id','hour'],how='inner')
        r_=mg.reached.values
        print(en,'tool rows',len(tool),'merged',len(mg),'reached agree',round((mg.reached==mg.mine_reached).mean(),3),
              '| tool',round(mg['shift'][r_].mean(),3),'mine',round(mg['mine_sh'][r_].mean(),3),
              '| corr shift',round(np.corrcoef(mg['shift'][r_],mg['mine_sh'][r_])[0,1],3),
              '| fact corr',round(np.corrcoef(mg.factual,mg.mine_fact)[0,1],4),'rmse',round(np.sqrt(np.mean((mg.factual-mg.mine_fact)**2)),2),
              'mean diff',round((mg.factual-mg.mine_fact).mean(),2))
```