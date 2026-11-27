`tr` has no `kind` column; that field is something `make_atoms` derives. I will look at the raw columns first, then run the same analysis of the insulin rule using whichever start-time and rate fields the raw records have.

**What success looks like:** a tier × source table showing insulin presence and mean rate.

```python
print(tr.columns.tolist()); print(tr.head(6).to_string())
ins_s = tr[tr.event=='insulin'].copy()
hcol = 'hour' if 'hour' in tr.columns else [c for c in tr.columns if 'hour' in c or 'time' in c][0]
ins_s['h'] = np.floor(ins_s[hcol].astype(float)+1e-6).astype(int)
gf_ = tr[tr.event=='glucose_finger'][['stay_id',hcol,'value']].copy(); gf_['h']=np.round(gf_[hcol].astype(float)).astype(int)
gl_ = tr[tr.event=='glucose_lab'][['stay_id',hcol,'value']].copy(); gl_['h']=np.round(gl_[hcol].astype(float)).astype(int)
grid_ = tr[['stay_id']].drop_duplicates().merge(pd.DataFrame({'h':np.arange(0,25)}), how='cross')
grid_ = grid_.merge(gf_[['stay_id','h','value']].rename(columns={'value':'gf'}).drop_duplicates(['stay_id','h']), on=['stay_id','h'], how='left')
grid_ = grid_.merge(gl_[['stay_id','h','value']].rename(columns={'value':'gl'}).drop_duplicates(['stay_id','h']), on=['stay_id','h'], how='left')
grid_ = grid_.merge(ins_s[['stay_id','h','rate']].drop_duplicates(['stay_id','h']), on=['stay_id','h'], how='left')
grid_['src'] = np.where(grid_.gf.notna(), 'finger', np.where(grid_.gl.notna(), 'lab', 'none'))
grid_['G'] = grid_.gf.fillna(grid_.gl)
grid_['tierG'] = pd.cut(grid_.G, [-1,150,180,220,250,1e9], right=False, labels=[0,1,2,3,4])
grid_['ins'] = grid_.rate.notna()
print(grid_.groupby(['src','tierG'], observed=True).agg(n=('ins','size'), p_ins=('ins','mean'), mean_rate=('rate','mean')).round(3).to_string())
print("none-src hours:", int((grid_.src=='none').sum()), " p_ins", round(float(grid_[grid_.src=='none'].ins.mean()),3), " both:", int((grid_.gf.notna()&grid_.gl.notna()).sum()))
```