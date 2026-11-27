The capture now lines up exactly with `tf_states_ds` (max difference 0.0). I have E for every bundle of 300 training stays: 16k bundles, collected in 34 s.

**The record format shows a gap in my notes.** At hour 0 of this stay there was no finger reading, only a lab glucose of 152, and insulin 0.5 was started at minute 5. So the insulin decision seems to use the latest glucose reading, lab included, not only `glucose_finger` as my notes say. The baseline −rbar(tier) has to be written for whichever reading the policy actually used.

**Next step:** check the insulin rule on the full training records. For each stay-hour I will find which glucose reading (finger, lab or none) was present, and see how insulin presence and the mean rate depend on its tier. I will also look at the `kind` codes, to see how infusion starts and ends are marked.

**What would confirm or refute the rule:**
- If insulin presence is nearly 0/1 by tier under both "finger" and "lab-only", the rule is "latest glucose at that hour".
- If insulin also appears in hours with no glucose reading, I need a carry-forward rule.

```python
print(tr.columns.tolist(), tr['kind'].unique()[:5])
print(tr.groupby(['event','kind']).size().to_string())
ins_s = tr[(tr.event=='insulin')&(tr.kind==0)][['stay_id','hour','rate']].copy(); ins_s['h']=np.floor(ins_s.hour+1e-6).astype(int)
gf_ = tr[tr.event=='glucose_finger'][['stay_id','hour','value']].copy(); gf_['h']=np.round(gf_.hour).astype(int)
gl_ = tr[tr.event=='glucose_lab'][['stay_id','hour','value']].copy(); gl_['h']=np.round(gl_.hour).astype(int)
grid_ = tr[['stay_id']].drop_duplicates().merge(pd.DataFrame({'h':np.arange(0,25)}), how='cross')
grid_ = grid_.merge(gf_[['stay_id','h','value']].rename(columns={'value':'gf'}), on=['stay_id','h'], how='left')
grid_ = grid_.merge(gl_[['stay_id','h','value']].rename(columns={'value':'gl'}).drop_duplicates(['stay_id','h']), on=['stay_id','h'], how='left')
grid_ = grid_.merge(ins_s[['stay_id','h','rate']], on=['stay_id','h'], how='left')
grid_['src'] = np.where(grid_.gf.notna(), 'finger', np.where(grid_.gl.notna(), 'lab', 'none'))
grid_['G'] = grid_.gf.fillna(grid_.gl)
grid_['tierG'] = pd.cut(grid_.G, [-1,150,180,220,250,1e9], right=False, labels=[0,1,2,3,4])
grid_['ins'] = grid_.rate.notna()
print(grid_.groupby(['src','tierG'], observed=True).agg(n=('ins','size'), p_ins=('ins','mean'), mean_rate=('rate','mean')).round(3).to_string())
print("both finger & lab same hour:", int((grid_.gf.notna()&grid_.gl.notna()).sum()))
```