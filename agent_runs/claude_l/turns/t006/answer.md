Insulin rates are multiples of 0.5 U/h with a long right tail (up to 14.5). Norepinephrine rates go in steps of 0.01 and are capped at 1.0 (32,919 records sit exactly at the cap), and NE dose is always rate × 4.8. So in both drugs dose is a fixed multiple of rate, and the two features carry the same information. One consequence for the evaluation: NE x2 pushes many rates past the training range (above 1.0), and insulin x2 pushes some past 5.0, where the bins saturate. In those regions only the linear z-score terms keep moving.

The crosstab hid the middle columns, so next I'll print insulin and map_arterial by hour. I'll also check, per stay, whether infusions run continuously from a start hour. That tells me whether the drug "episodes" are on-off or titrated hour by hour.

```python
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(pd.crosstab(tr.ih, tr.event)[['insulin','map_arterial','norepinephrine','glucose_lab']].T)
# per stay: first/last insulin hour, count, contiguity
def runinfo(df):
    h = np.sort(df.ih.values)
    return pd.Series({'first':h.min(),'last':h.max(),'n':len(h),'contig': (h.max()-h.min()+1)==len(h)})
ri = ins.assign(ih=np.floor(ins.hour).astype(int)).groupby('stay_id').apply(runinfo)
print(ri.describe()); print("contiguous frac", ri.contig.mean())
print(ri['first'].value_counts().sort_index().head(30).to_dict())
rn = ne.assign(ih=np.floor(ne.hour).astype(int)).groupby('stay_id').apply(runinfo)
print(rn.describe()); print("contiguous frac NE", rn.contig.mean())
print(rn['first'].value_counts().sort_index().head(30).to_dict())
```