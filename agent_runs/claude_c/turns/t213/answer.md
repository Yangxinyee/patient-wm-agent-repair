`fwd_2` holds the original model's factual output. I'll use it as the baseline, after checking its mean NLL is 1.63461. Then I'll split the per-reading NLL change by event and by time since the last drug record started, for three runs: head-0 ×4 (pass 22), attn.out ×3 (pass 12) and u=0 (pass 15). Cost concentrated right after drug records would point to double-counting of the systematic dose association; cost spread evenly would point to disruption of reading processing.

```python
fo_ = pd.read_csv('/work/fwd_2/factual_original.csv'); print('orig mean', round(fo_.nll.mean(),5))
dr_ = tu[tu.event.isin(['insulin','norepinephrine'])][['stay_id','hour']].sort_values('hour')
def since(df):
    out=[]
    for sid, g in df.groupby('stay_id'):
        t = np.sort(dr_[dr_.stay_id==sid].hour.values)
        idx = np.searchsorted(t, g.hour.values, side='left')-1
        out.append(pd.Series(np.where(idx>=0, g.hour.values - t[np.clip(idx,0,None)] if len(t) else 99, 99), index=g.index))
    return pd.concat(out)
fo_['since']=since(fo_); fo_['sb']=pd.cut(fo_.since,[-1,1.5,4,12,1e9],labels=['<1.5','1.5-4','4-12','>12/none'])
for k_,fn_ in [(22,'h0x4'),(12,'attnx3'),(15,'u0')]:
    e_=pd.read_csv('/work/fwd_%d/factual_edited.csv'%k_); dd_=fo_.copy(); dd_['d']=e_.nll.values-fo_.nll.values
    print(fn_, round(dd_.d.mean(),4)); print(dd_.pivot_table(index='event',columns='sb',values='d',aggfunc='mean').round(4).to_string())
print(fo_.groupby('sb').size().to_dict())
```