The counterfactual runner works and takes 0.09 s per stay run.

Two findings matter:
1. Over the 28 bundles after hour 12, the replica drifts from the tool's states by up to 0.2 in `out`, as per-step precision errors of about 0.01 accumulate. That is acceptable, because counterfactual and factual replica runs share the same numerical path.
2. Doubling every insulin dose from hour 12 changes `out` by at most 0.028. The state hardly encodes insulin dose, consistent with the model's weak dose response.

Whether a readout can recover that weak signal robustly is now the central question. I will build the full pipeline on the 100 sel1 stays for insulin ×2:
- factual and counterfactual replica states;
- head query q with common z draws at every glucose reading after hour 12;
- the model's predicted-mean change;
- the structural target: the direct effect given the latest reading, Δ = −2·Σ 0.6^(t−m)·Δx_m, with insulin compartment x.

Then I will report the mean structural and model shifts and the size of Δq relative to q's natural spread. A small model shift relative to the structural one confirms the diagnosis. The Δq/spread ratio tells me whether a readout can work without amplifying noise.

```python
code_cf = r'''
def drug_hist(s,drug):
    dd={}
    for (mnt,evs) in BUN[s]:
        for (ev,kd,row) in evs:
            if ev==drug and kd=='start': dd[int(round((mnt-5)/60))]=float(row.rate)
    return dd
GLU=('glucose_finger','glucose_lab'); MAPC=('map_arterial','map_cuff')
def struct_delta(s,minute,codeset,f,start_h=12):
    glu=codeset==GLU; drug='insulin' if glu else 'norepinephrine'; a=0.6 if glu else 0.39; gain=-2.0 if glu else 3.3
    dd=drug_hist(s,drug); tau=None
    for (mnt,evs) in BUN[s]:
        if mnt>=minute: break
        if any(ev in codeset and kd=='world' for (ev,kd,row) in evs): tau=mnt
    t=int(round(minute/60)); th=int(round(tau/60)) if tau is not None else -99
    dx=0.0; tot=0.0
    for m in range(start_h+1,t+1):
        dx=a*dx+(f-1)*dd.get(m-1,0.0)*(m-1>=start_h)
        if m>th: tot=tot*a+dx
        else: tot=0.0
    # tot accumulates sum_{m>th} a^{t-m} dx_m
    return gain*tot
def heads_on_states(O,H,s,rt,eps):
    # rt: reading-table rows of stay s (b>=1); eps: (n,4,32) fixed draws
    b=torch.tensor(rt['b'].values); dt=torch.tensor(rt['dt'].values,dtype=torch.float64)
    out=O[b-1]; h=H[b-1]
    hf=h*torch.exp(-dt[:,None,None]*torch.exp(HT('core.flow.a')))
    ctx=out+Hln(hf.mean(1),'core.out_norm')+Hlin(Fn.gelu(Hlin(torch.log(dt)[:,None]/10,'gap.0')),'gap.2')
    zp=Hlin(Fn.gelu(Hlin(ctx,'z_prior.0')),'z_prior.2')
    z=zp[:,None,:32]+torch.exp(zp[:,None,32:])*eps
    q=Hln(ctx[:,None,:]+Hlin(z,'z_head')+HT('type_emb.weight')[1],'query_norm')
    f,lb,cid=last_feats(rt); wc=with_code(q,f,lb,cid); P=value_probs(wc,lb,cid)
    cen=torch.tensor(np.stack([CEN[c] for c in rt['event']]))
    return q,wc,P,(P[:,1:]*cen).sum(1)/P[:,1:].sum(1)
'''
open('/work/cf.py','w').write(code_cf); exec(code_cf)
sels=sorted(set(int(x) for x in S['stay_id']))
RTs={s:RT[(RT.stay_id==s)&(RT.b>=1)].reset_index(drop=True) for s in sels}
rows=[]
for s in sels:
    rt=RTs[s]; Of,Hf,k0=run_cf(s,{}); Oc,Hc,_=run_cf(s,{'insulin':2.0})
    m=(rt['b']>k0)&rt['event'].isin(GLU); rt=rt[m].reset_index(drop=True)
    if len(rt)==0: continue
    eps=torch.randn(len(rt),4,32,generator=torch.Generator().manual_seed(s%100000),dtype=torch.float64)
    qf,_,_,mf=heads_on_states(Of,Hf,s,rt,eps); qc,_,_,mc=heads_on_states(Oc,Hc,s,rt,eps)
    for i in range(len(rt)):
        rows.append(dict(s=s,minute=rt.minute[i],ev=rt.event[i],dmod=float(mc[i]-mf[i]),dstr=struct_delta(s,rt.minute[i],GLU,2.0),dq=float((qc[i]-qf[i]).mean(0).norm()),qsd=float(qf[i].std())))
D2=pd.DataFrame(rows)
print(len(D2), 'mean struct',round(D2.dstr.mean(),2),'mean model',round(D2.dmod.mean(),2),'corr',round(np.corrcoef(D2.dstr,D2.dmod)[0,1],2))
print('|dq| med',round(D2.dq.median(),4),'max',round(D2.dq.max(),3))
```