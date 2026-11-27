Every type_emb row is trained, with norms of 12–17, so the query type for glucose cannot be read off the norms. I will brute-force it over the 15 rows. last_status has a padding row 0 and a trained row 1. last_numeric uses all 8 columns, so its layout has to be inferred. The glucose_finger bin centres run from 73 to 271 mg/dL.

Next I build the local value head from the stored states:
- the context and the view, using the prior mean of z;
- the query for each candidate type;
- the register terms for the last same-code value;
- the value MLP plus the transition table, giving the predicted mean.

I compare it with the model's predicted means for glucose_finger readings, over all 15 types, several last_numeric layouts and both last_status indices. A combination with errors well under 1 mg/dL, allowing for z-sampling noise, would validate the head. If every combination is off by several mg/dL, a structural detail is still wrong.

```python
key_k = {}
for sid_ in np.unique(S_['stay_id']):
    idx_=np.where(S_['stay_id']==sid_)[0]; key_k[sid_]=(idx_, S_['hour'][idx_])
recs = tr[tr.stay_id.isin(list(key_k.keys()))]
def last_same(sid_, ev_, hr_):
    r_=recs[(recs.stay_id==sid_)&(recs.event==ev_)&(recs.hour<hr_-1e-9)]
    return (None,None) if len(r_)==0 else (float(r_.value.iloc[-1]), float(r_.hour.iloc[-1]))
def head_inputs(row):
    idx_,hrs_=key_k[row.stay_id]; k=int(np.where(np.isclose(hrs_,row.hour))[0][0]); i0=idx_[k-1]
    dt=(hrs_[k]-hrs_[k-1])*3600; lv,lt=last_same(row.stay_id,row.event,row.hour)
    return i0, dt, lv, (None if lt is None else (row.hour-lt)*3600)
fs = fac[fac.event=='glucose_finger'].iloc[:250]
HI = [head_inputs(r) for r in fs.itertuples()]
def pred_mean(i0, dt, lv, age, code_name, typ, layout, lstat, zmode='mean'):
    out_=torch.tensor(S_['x'][i0,:192]); h_=torch.tensor(S_['x'][i0,192:960]).reshape(4,192)
    hf=h_*torch.exp(-dt*torch.exp(fa)); ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
    pr=MLP(ctx,'z_prior','gelu'); zm=pr[:32]
    view=ctx+L_(zm,'z_head'); q=LN(view+Wo['type_emb.weight'][typ],'query_norm')
    c_=CODE[code_name]; wc=q+Wo['embed.code.weight'][c_]
    num=torch.zeros(8); lb=0
    if lv is not None:
        zl=zsc(code_name,'value',lv); lb=fbin(code_name,'value',lv)
        for pos,val in zip(layout,[1.0, math.log(1+age)/10, zl, 1.0]):
            if pos is not None: num[pos]=val
    wc=wc+L_(num,'last_numeric')+Wo['embed.bins.value.weight'][lb]+(Wo['last_status.weight'][lstat] if lv is not None else 0)
    lg=MLP(wc,'attr_heads.value_bin','gelu')+Wo['embed_transition.value_bin.weight'][c_*17+lb]
    p=torch.softmax(lg,0); cen=torch.tensor(VOCAB['value_bin_centres'][code_name],dtype=torch.float32)
    return float((p[1:]*cen).sum())
layouts = {'A[has,age,z,hasv@3]':(0,1,2,3), 'B[has,age,z,-]':(0,1,2,None), 'C[has,age,z@2,hasv@4]':(0,1,2,4)}
obs_pm = fs.predicted_mean.values
for lname,lay in layouts.items():
    for lstat in [0,1]:
        errs_t=[]
        for typ in range(15):
            pm=np.array([pred_mean(i0,dt,lv,age,'glucose_finger',typ,lay,lstat) for (i0,dt,lv,age) in HI[:60]])
            errs_t.append(np.median(np.abs(pm-obs_pm[:60])))
        j=int(np.argmin(errs_t)); print(lname, 'lstat',lstat, 'best type',j, 'median err', round(errs_t[j],2), 'second', round(sorted(errs_t)[1],2))
```