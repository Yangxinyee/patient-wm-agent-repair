The counterfactual pipeline works, and it shows the gap directly. With readings held as recorded, doubling insulin from hour 12 has a structural direct effect averaging −5.78 mg/dL over 1,150 glucose readings. The model's own direct response averages only −0.78, about 13%, with per-reading correlation 0.49. So the model sees the dose but under-weights it about 7-fold.

The head query moves by only about 0.1 in norm. A readout therefore needs weight norm of about 5 or more along that direction, and its noise on recorded data will decide feasibility.

I will now generate the full training set on the 100 stays:
- Factual replica runs from hours 6, 12 and 16.
- Insulin edits ×0, ×½, ×2, ×3 from hour 12, and ×0, ×2 from hours 6 and 16.
- The same edits for norepinephrine.
- For each reading after the edit, I store per-draw `with_code`, per-draw logits, the structural target and the model's shift.

The code goes to /work/gen.py so it survives restarts.

```python
gen_code = r'''
def logits_draws(wc,lb,cid):
    return Hlin(Fn.gelu(Hlin(wc,'attr_heads.value_bin.0')),'attr_heads.value_bin.2')+HT('embed_transition.value_bin.weight')[cid*17+lb][:,None,:]
def heads_full(O,H,rt,eps):
    b=torch.tensor(rt['b'].values); dt=torch.tensor(rt['dt'].values,dtype=torch.float64)
    out=O[b-1]; h=H[b-1]; hf=h*torch.exp(-dt[:,None,None]*torch.exp(HT('core.flow.a')))
    ctx=out+Hln(hf.mean(1),'core.out_norm')+Hlin(Fn.gelu(Hlin(torch.log(dt)[:,None]/10,'gap.0')),'gap.2')
    zp=Hlin(Fn.gelu(Hlin(ctx,'z_prior.0')),'z_prior.2'); z=zp[:,None,:32]+torch.exp(zp[:,None,32:])*eps
    q=Hln(ctx[:,None,:]+Hlin(z,'z_head')+HT('type_emb.weight')[1],'query_norm')
    f,lb,cid=last_feats(rt); wc=with_code(q,f,lb,cid); return wc.float(), logits_draws(wc,lb,cid).float()
CONDS=[('insulin',f_,sh) for sh in (12,) for f_ in (0.0,0.5,2.0,3.0)]+[('insulin',f_,sh) for sh in (6,16) for f_ in (0.0,2.0)]
CONDS+= [('norepinephrine',f_,sh) for (d_,f_,sh) in CONDS]
def gen_stay(s):
    rt_all=RTs[s]; res=[]
    if len(rt_all)==0: return res
    eps=torch.randn(len(rt_all),4,32,generator=torch.Generator().manual_seed(s%100000),dtype=torch.float64)
    fact={}
    for sh in (6,12,16):
        Of,Hf,k0=run_cf(s,{},k_from_min=sh*60); fact[sh]=(k0,)+heads_full(Of,Hf,rt_all,eps)
    for (drug,f_,sh) in CONDS:
        cs=GLU if drug=='insulin' else MAPC
        k0,wcf,lgf=fact[sh]; m=((rt_all['b']>k0)&rt_all['event'].isin(cs)).values
        if m.sum()==0: continue
        Oc,Hc,_=run_cf(s,{drug:f_},k_from_min=sh*60); wcc,lgc=heads_full(Oc,Hc,rt_all,eps)
        idx=np.where(m)[0]
        st=np.array([struct_delta(s,rt_all.minute[i],cs,f_,start_h=sh) for i in idx])
        res.append(dict(s=s,drug=drug,f=f_,sh=sh,idx=idx,st=st,wc=wcc[idx],lg=lgc[idx],lgf=lgf[idx],ev=rt_all.event.values[idx],minute=rt_all.minute.values[idx]))
    res.append(dict(s=s,drug='fact',f=1.0,sh=12,idx=np.arange(len(rt_all)),wc=fact[12][1],lg=fact[12][2],ev=rt_all.event.values,minute=rt_all.minute.values))
    return res
'''
open('/work/gen.py','w').write(gen_code); exec(gen_code)
t0=time.time(); GEN=[]
for s in sels: GEN+=gen_stay(s)
print('time',round(time.time()-t0,1),'items',len(GEN))
def pmean(lg,ev):
    P=torch.softmax(lg.double(),-1).mean(1); cen=torch.tensor(np.stack([CEN[c] for c in ev]))
    return ((P[:,1:]*cen).sum(1)/P[:,1:].sum(1)).numpy()
summ=[]
for it in GEN:
    if it['drug']=='fact': continue
    dm=pmean(it['lg'],it['ev'])-pmean(it['lgf'],it['ev'])
    summ.append(dict(drug=it['drug'],f=it['f'],sh=it['sh'],n=len(dm),st=it['st'].sum(),dm=dm.sum()))
sm=pd.DataFrame(summ).groupby(['drug','f','sh']).sum(); sm['st']/=sm['n']; sm['dm']/=sm['n']
print(sm.round(2).to_string())
```