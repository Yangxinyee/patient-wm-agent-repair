The z pathway barely matters: setting z to zero gives about the same error. So the remaining ~7 mg/dL glucose error must come from the register or transition features, or from how states align with prediction times. To find which, I'll break the residual (reproduced minus CSV mean) down by event, has-last, previous-bundle type, gap length and atom count. If a component is wrong, the error should cluster in one subgroup, for example readings whose previous bundle was an insulin bundle, or long gaps.

```python
res = predict(FR,typ=1,gap_scale=10.0)
FR['e'] = res[0]-FR['predicted_mean'].values; FR['dn']=res[1]-FR['nll'].values
st_idx = st.set_index('row')
prev = FR['prev_row'].astype(int).values
FR['prev_ins'] = S3['has_insulin'][prev]; FR['prev_ne']=S3['has_norepinephrine'][prev]; FR['prev_n']=S3['n_atoms'][prev]
FR['dtm'] = FR['minute']-FR['prev_min']; FR['has']=FR['last_val'].notna()
def tab(col):
    t = FR.groupby(['event',col])['e'].agg(['mean',lambda x: np.sqrt((x**2).mean()),'size']); t.columns=['bias','rms','n']; return t.round(2)
print(tab('has').to_string()); print(tab('prev_ins').to_string())
print(FR.groupby(['event',pd.cut(FR['dtm'],[0,5,30,55,60,120,1e5])])['e'].agg(['mean','size']).round(2).to_string())
```