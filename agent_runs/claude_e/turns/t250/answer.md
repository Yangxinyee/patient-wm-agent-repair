Each per-reading file has stay, hour, event, observed, predicted mean and NLL. I will load all four factual "both" runs from the last gain checks (fwd_13, 14, 15 and 17). Each one is identified by its stay range and its overall cost, which should match my notes: tu0 0.00427 and tu2 0.00100 at −2.6, f3 0.00284 and tu1 0.00369 at −2.5. Then I scale the −2.5 batches by 1.12 and resample evaluation sets of several sizes to estimate the risk of exceeding 0.005.

```python
rows_=[]
for k in [13,14,15,17]:
    try: o_,e_=stay_cost(k)
    except Exception as ex: print(k,'ERR',ex); continue
    assert len(o_)==len(e_) and (o_.stay_id.values==e_.stay_id.values).all()
    d_=e_.nll.values-o_.nll.values
    print(k,'stays',o_.stay_id.min(),o_.stay_id.max(),'n',len(o_),'diff',d_.mean().round(5))
    t_=pd.DataFrame({'stay':o_.stay_id.values,'d':d_}).groupby('stay').agg(sd=('d','sum'),n=('d','size')).reset_index(); t_['fwd']=k; rows_.append(t_)
PS=pd.concat(rows_,ignore_index=True)
PS['sd26']=np.where(PS.fwd.isin([14,15]),PS.sd*1.12,PS.sd)
print('pooled per-reading cost at -2.6 est',(PS.sd26.sum()/PS.n.sum()).round(5),' at -2.5 est',((np.where(PS.fwd.isin([13,17]),PS.sd/1.12,PS.sd)).sum()/PS.n.sum()).round(5))
rng=np.random.default_rng(0)
for lab,col in [('-2.6','sd26'),('-2.5',None)]:
    v=PS.sd26.values if col else np.where(PS.fwd.isin([13,17]),PS.sd/1.12,PS.sd); nn=PS.n.values
    for N in [100,200,400,1000]:
        ii=rng.integers(0,len(PS),size=(4000,N)); c=v[ii].sum(1)/nn[ii].sum(1)
        print(lab,'N',N,'mean',c.mean().round(5),'q95',np.quantile(c,.95).round(5),'P>.005',(c>.005).mean().round(4))
```