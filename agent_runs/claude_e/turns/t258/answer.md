Training batch 3 at −2.6 costs only +0.00104. Over all eight batches (800 stays), −2.6 averages about 0.0027 per reading. The batch values run from 0.0010 to 0.0043, and the mean is uncertain to about ±0.0005. That is lower than the tuning-only estimate.

Before settling on −2.6, I will redo the stay-level resampling on all 800 stays. −2.5 is approximated by scaling the −2.6 costs down by 1.12, which is a rough factor. I will also record the batch results in the notebook.

If the chance of exceeding 0.005 at −2.6 is below about 1% for evaluation sets of 400+ stays, I keep −2.6. It is about 1 point closer on x2 than −2.5 (about 99% versus 95% of the truth estimate) and slightly closer on x1/2. If not, I go back to −2.5.

```python
parts=[]
for k in [13,14,15,17,18,19,20,21]:
    o_,e_=stay_cost(k); d_=e_.nll.values-o_.nll.values
    t_=pd.DataFrame({'stay':o_.stay_id.values,'d':d_}).groupby('stay').agg(sd=('d','sum'),n=('d','size')).reset_index(); t_['fwd']=k; parts.append(t_)
PS8=pd.concat(parts,ignore_index=True)
v26=np.where(PS8.fwd.isin([14,15]),PS8.sd*1.12,PS8.sd); v25=v26/1.12; nn=PS8.n.values
print('batch costs -2.6:',{k:round(float(v26[PS8.fwd==k].sum()/nn[PS8.fwd==k].sum()),5) for k in [13,14,15,17,18,19,20,21]})
print('pooled -2.6',round(v26.sum()/nn.sum(),5),'-2.5',round(v25.sum()/nn.sum(),5))
rng=np.random.default_rng(1)
for lab,v in [('-2.6',v26),('-2.55',(v26+v25)/2),('-2.5',v25)]:
    out_=[]
    for N in [100,200,400,1000]:
        ii=rng.integers(0,len(PS8),size=(6000,N)); c=v[ii].sum(1)/nn[ii].sum(1); out_.append((N,round(float(np.quantile(c,.99)),5),round(float((c>.005).mean()),4)))
    print(lab,out_)
with open('/work/notes.md','a') as f:
    f.write("""- fwd_19/20/21 (train batches 1-3 at -2.6): +.00261, +.00368, +.00104. 8 batches (800 stays) at -2.6: .00428,.00317*,.00413*,.00100,.00164,.00261,.00368,.00104 (*scaled from -2.5 x1.12). Mean ~.0027.
""")
```