The measured cost settles it. At full gain the head readout costs 0.014 nats per reading for insulin and 0.026 for norepinephrine, against a 0.005 budget, and doubling the gain costs 0.16. Reading the model's weak native dose encoding at the head is not viable.

The alternative is to strengthen the dose pathway at its source. The `z_dose`/`z_rate` columns of `embed.numeric` are used only by drug atoms. Scaling them by λ should scale the state's dose response, and with it the model's own direct effect, which is currently 10–15%.

The factual side effect should then be mostly a function of the last glucose bin, which I can cancel per (code, last bin) in `embed_transition`. The leftover deviation part points the right way.

I will test this with the replica at λ = 3 and 6. For each I will run factual and insulin ×2 counterfactual states and report the direct-effect ratio to structural, plus the factual NLL change with and without group cancellation. If the ratio scales roughly with λ and the cancelled NLL change stays small, this is the edit to pursue.

```python
c_=drug_atom.__code__; print('drug_atom refs',[n for n in c_.co_names if n in globals()][:10])
NW0=Wt['embed.numeric.weight'].clone(); NW0W=W['embed.numeric.weight'].clone() if 'embed.numeric.weight' in W else None
def set_amp(lam):
    for D_,base in [(Wt,NW0),(W,NW0W)]:
        if base is None: continue
        nw=base.clone().to(D_['embed.numeric.weight'].dtype); nw[:,2]*=lam; nw[:,4]*=lam; D_['embed.numeric.weight']=nw
    CACHE.clear()
def first_drug_min(s):
    return min([m for (m,evs) in BUN[s] if any(a[0] in ('insulin','norepinephrine') for a in evs)]+[10**6])
def amp_eval(lam,stays):
    set_amp(lam); r=[]
    for s in stays:
        rt=RTs[s]; k_m=first_drug_min(s)
        if k_m>=10**6: continue
        eps=torch.randn(len(rt),4,32,generator=torch.Generator().manual_seed(s%100000),dtype=torch.float64)
        Of,Hf,k0=run_cf(s,{},k_from_min=k_m); wcf,lgf=heads_full(Of,Hf,rt,eps)
        Oc,Hc,k1=run_cf(s,{'insulin':2.0},k_from_min=720); wcc,lgc=heads_full(Oc,Hc,rt,eps)
        r.append((s,k0,k1,lgf,lgc))
    set_amp(1.0); return r
t0=time.time(); A3=amp_eval(3.0,sels[:40]); print('t',round(time.time()-t0,1))
```