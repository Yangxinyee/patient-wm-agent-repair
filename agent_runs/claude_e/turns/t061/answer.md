Type 1 is best for every event, so a per-event type is not the bug. Readings with no previous value are still off, so the error sits in the base view and query computation, not the register.

The remaining candidates:
- the order of the prior's output (mean first or log-sd first);
- the activation (exact GELU, tanh-GELU, SiLU);
- the unit of the gap input (seconds, minutes or hours, with or without /10).

I will grid these jointly, keeping type 1 and layout B. A combination giving MAE well under 1 identifies the bug. If nothing goes below about 3, the doc's formula is missing a term, and I will look for unused tensors, such as `embed.categorical.unit` being added in the head.

```python
from scipy.special import expit
ACTS={'gelu':gelu,'gelu_tanh':lambda x:0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))),'silu':lambda x:x*expit(x)}
def full_mean(act,gin,zfirst):
    a=ACTS[act]
    g=L(a(L(gin[:,None],'gap.0')),'gap.2'); ctx=out+mm+g
    pr=L(a(L(ctx,'z_prior.0')),'z_prior.2'); z=pr[:,:32] if zfirst else pr[:,32:]
    V=ctx+L(z,'z_head')
    q=lnorm(V+Wt['type_emb.weight'][1],Wt['query_norm.weight'],Wt['query_norm.bias'])
    wc=q+Wt['embed.code.weight'][code_ids]+L(R,'last_numeric')+Wt['embed.bins.value.weight'][lb]
    lg=L(a(L(wc,'attr_heads.value_bin.0')),'attr_heads.value_bin.2')+Wt['embed_transition.value_bin.weight'][code_ids*17+lb]
    p=np.exp(lg-lg.max(1,keepdims=True)); p/=p.sum(1,keepdims=True); return (p[:,1:]*cen).sum(1)
gins={'ln_s/10':np.log(dtv)/10,'ln_min/10':np.log(dtv/60)/10,'ln_h':np.log(dtv/3600),'ln1p_h':np.log1p(dtv/3600),'ln_s':np.log(dtv),'s/1e4':dtv/1e4}
res_=[]
for act in ACTS:
    for gn,gi in gins.items():
        for zf in [True,False]:
            res_.append((round(np.abs(full_mean(act,gi,zf)-tool).mean(),3),act,gn,zf))
res_.sort(); print(res_[:8])
```