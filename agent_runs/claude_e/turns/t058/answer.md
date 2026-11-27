The error depends on the event. Finger glucose is worst (MAE 5.9, bias −1.7) and arterial MAP best (1.5). Readings with a previous value are worse than those without. The no-decay variant is worse and gap(log dt) without the /10 is far worse, so time flow and the log(dt)/10 gap look right.

Two causes remain likely:
1. **z handling.** The tool averages probabilities over 4 prior draws, and I used the prior mean. A broad prior would make that matter.
2. **Register age units.** Seconds or hours inside log(1 + age)/10.

This run checks the prior's spread and tries averaging over 64 draws, with age in seconds or in hours. If draws fix it, MAE should drop a lot. If the age unit matters, the readings that have a previous value should improve most.

```python
st=Xs[kv]; out=st[:,:192]; hs=st[:,192:960].reshape(-1,4,192)
hf=hs*np.exp(-dtv[:,None,None]*np.exp(Wt['core.flow.a'])[None,None,:]); mm=lnorm(hf.mean(1),Wt['core.out_norm.weight'],Wt['core.out_norm.bias'])
g10=L(gelu(L(np.log(dtv)[:,None]/10,'gap.0')),'gap.2'); CTX=out+mm+g10
PR=L(gelu(L(CTX,'z_prior.0')),'z_prior.2'); zmu,zls=PR[:,:32],PR[:,32:]
print("prior logsd: mean %.2f min %.2f max %.2f"%(zls.mean(),zls.min(),zls.max()))
def regs2(F,layout,age_unit=1.0):
    R,lb=regs(F,layout); has=F.prev_val.notna().values
    age=np.where(has,(F.hour-F.prev_hr).values*3600/age_unit,0); R[:,1]=np.log1p(age)/10; return R,lb
def head_p(F,V,T,R,lb,code_ids):
    q=lnorm(V+Wt['type_emb.weight'][T],Wt['query_norm.weight'],Wt['query_norm.bias'])
    wc=q+Wt['embed.code.weight'][code_ids]+L(R,'last_numeric')+Wt['embed.bins.value.weight'][lb]
    lg=L(gelu(L(wc,'attr_heads.value_bin.0')),'attr_heads.value_bin.2')+Wt['embed_transition.value_bin.weight'][code_ids*17+lb]
    p=np.exp(lg-lg.max(1,keepdims=True)); return p/p.sum(1,keepdims=True)
cen=np.stack([CEN[e] for e in F.event]); tool=F.predicted_mean.values
rng=np.random.default_rng(0)
for au in [1.0,3600.0]:
    R,lb=regs2(F,'B',au)
    p0=head_p(F,CTX+L(zmu,'z_head'),1,R,lb,code_ids); pmean=(p0[:,1:]*cen).sum(1)
    pacc=0
    for _ in range(64): pacc=pacc+head_p(F,CTX+L(zmu+np.exp(zls)*rng.standard_normal(zmu.shape),'z_head'),1,R,lb,code_ids)
    pmc=(pacc[:,1:]/64*cen).sum(1)
    print("age unit",au,"| MAE z=mean %.3f | MAE z MC64 %.3f"%(np.abs(pmean-tool).mean(),np.abs(pmc-tool).mean()))
```