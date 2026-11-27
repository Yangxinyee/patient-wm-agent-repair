The atom table gives each atom's bundle row, event, kind (0 = start, 2 = END) and rate, so bundles can be classified directly.

On the 300-stay set I'll run `tf_loop` to get hf, E and z for every bundle, recompute the prelude internals, and tabulate the u-token attention weight per head for each bundle type: insulin start, NE start without insulin, END only, and no drug. A head with weight near 1 on u at both insulin and NE bundles would serve as the read channel for both drugs.

```python
def classify(ds_):
    a=ds_['at']; N=len(ds_['e'])
    def agg(mask):
        s=np.zeros(N); 
        if mask.any(): np.add.at(s, a.loc[mask,'row'].values, np.nan_to_num(a.loc[mask,'rate'].values) if True else 1)
        return s
    def has(mask):
        h=np.zeros(N,bool); h[a.loc[mask,'row'].values]=True; return h
    ins=(a.event=='insulin')&(a.kind==0); ne=(a.event=='norepinephrine')&(a.kind==0); en=(a.kind==2)
    meas=a.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])
    return dict(hI=has(ins), rI=agg(ins), hN=has(ne), rN=agg(ne), hEnd=has(en), hMeas=has(meas))
t0=time.time()
rE=tf_loop(PO, dsE, keep=('E','hf','hn')); stE=rE[2]
cE=classify(dsE)
prE=prelude(PO, stE['hf'], dsE['e'], dsE['u'], stE['z'])
print('N', len(dsE['e']), '%.1fs'%(time.time()-t0))
att=prE['att']  # (N,H,7,7); u token index 5
ty={'ins':cE['hI'], 'neOnly':cE['hN']&~cE['hI'], 'endOnly':cE['hEnd']&~cE['hI']&~cE['hN'], 'meas':cE['hMeas']&~cE['hEnd']&~cE['hI']&~cE['hN']}
for k,m in ty.items():
    w=att[torch.as_tensor(m)][:,:,:4,5]  # slot queries -> u key
    print(k, int(m.sum()), 'u-weight per head (mean,min over slots):', [(round(float(w[:,h].mean()),3), round(float(w[:,h].min()),3)) for h in range(4)])
```