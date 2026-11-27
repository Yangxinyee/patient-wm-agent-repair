The cost estimates are noisy. The stay-clustered standard error is about 0.0011 for numeric ×5 and 0.0017 for the combination. To keep a safe margin on unseen stays, an estimated cost above about 0.0035 is too risky. That rules out head-0 ×2.5 and leaves little room for head-0 at all, given its poor ratio of about 100 mg/dL of readout per nat.

Before settling, I want to know whether numeric ×5's own cost can be cut. If it sits mainly after norepinephrine records (map_arterial) or at lab glucose, a drug-specific variant might keep the insulin gain more cheaply. I'll break pass 24's cost down by event and by hours since the last insulin and the last norepinephrine record.

```python
o24=pd.read_csv('/work/fwd_24/factual_original.csv'); e24=pd.read_csv('/work/fwd_24/factual_edited.csv'); o24['d']=e24.nll.values-o24.nll.values
def since_drug(df, drug):
    t_all = tu[tu.event==drug].groupby('stay_id').hour.apply(np.sort).to_dict(); out=np.full(len(df), 99.0)
    for i_,(s_,h_) in enumerate(zip(df.stay_id.values, df.hour.values)):
        t=t_all.get(s_); 
        if t is not None:
            j=np.searchsorted(t,h_)-1
            if j>=0: out[i_]=h_-t[j]
    return out
o24['si']=since_drug(o24,'insulin'); o24['sn']=since_drug(o24,'norepinephrine')
cut_=lambda s: pd.cut(s,[-1,1.5,4,1e9],labels=['<1.5','1.5-4','>4/none'])
print('by since-insulin\n', o24.pivot_table(index='event',columns=cut_(o24.si),values='d',aggfunc='mean').round(4).to_string())
print('by since-NE\n', o24.pivot_table(index='event',columns=cut_(o24.sn),values='d',aggfunc='mean').round(4).to_string())
print(o24.groupby(cut_(o24.si)).size().to_dict(), o24.groupby(cut_(o24.sn)).size().to_dict())
```