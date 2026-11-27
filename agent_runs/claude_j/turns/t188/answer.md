Gating along the exactly q-invariant direction 1/γ fails. Glucose_finger (−2.48) and glucose_lab (+3.12) fall on opposite sides of the MAP codes (+0.86, +0.53). The last-value terms, z_last_value especially (−2.57 per z unit), also swing a lot along that direction.

One fact helps: at a given state q is identical for every code. So code separation depends only on the code embeddings and last-value terms. What I need is a direction w with large separation between target and non-target codes relative to the within-code spread of `with_code` (a Mahalanobis or LDA ratio).

To measure that I need a clean head replica. I will build `with_code` and value logits for every recorded reading in the fwd_3 states, using 4 prior z draws, and validate the predicted means against the tool's factual output (fwd_2). I will also check whether the mean renormalises over bins 1..16. If the MAE stays around 1, the head replica is good enough to fit the readout and gate.

```python
import pandas as pd
fa2=pd.read_csv('/work/fwd_2/factual_original.csv')
RT=build_reading_table(S, fa2)
Xd=torch.tensor(S['x'],dtype=torch.float64)
def q_draws(RT_,Xd_,nd=4,seed=0):
    dt=torch.tensor(RT_['dt'].values,dtype=torch.float64); prow=torch.tensor(RT_['prow'].values)
    out,mh,gv,ctx,zp=view_parts(Xd_,prow,dt)
    g=torch.Generator().manual_seed(seed); eps=torch.randn(len(RT_),nd,32,generator=g,dtype=torch.float64)
    z=zp[:,None,:32]+torch.exp(zp[:,None,32:])*eps
    view=ctx[:,None,:]+Hlin(z,'z_head')
    return Hln(view+HT('type_emb.weight')[1],'query_norm')
def last_feats(RT_):
    codes=RT_['event'].values; n=len(RT_)
    f=np.zeros((n,8)); f[:,0]=RT_['has_last'].values; f[:,3]=np.log1p(RT_['age'].values)/10
    zl=[zs_(c,v) if h else 0.0 for c,v,h in zip(codes,RT_['last_val'].values,RT_['has_last'].values)]
    f[:,6]=zl
    lb=[fb_(c,v) if h else 0 for c,v,h in zip(codes,RT_['last_val'].values,RT_['has_last'].values)]
    cid=[CID[c] for c in codes]
    return torch.tensor(f),torch.tensor(lb),torch.tensor(cid)
def with_code(q,f,lb,cid):
    add=HT('embed.code.weight')[cid]+Hlin(f,'last_numeric')+HT('embed.bins.value.weight')[lb]
    return q+add[:,None,:]
def value_probs(wc,lb,cid):
    lg=Hlin(Fn.gelu(Hlin(wc,'attr_heads.value_bin.0')),'attr_heads.value_bin.2')+HT('embed_transition.value_bin.weight')[cid*17+lb][:,None,:]
    return torch.softmax(lg,-1).mean(1)
Q4=q_draws(RT,Xd); fL,lbL,cidL=last_feats(RT)
P=value_probs(with_code(Q4,fL,lbL,cidL),lbL,cidL).numpy()
cenM=np.stack([CEN[c] for c in RT['event']])
m_raw=(P[:,1:]*cenM).sum(1); m_ren=m_raw/P[:,1:].sum(1)
ok=RT['predicted_mean'].notna()&(RT['has_last']==1)
for nm,m in [('raw',m_raw),('renorm',m_ren)]:
    print(nm, 'MAE', round(float(np.abs(m-RT['predicted_mean'].values)[ok].mean()),3), 'p0 mean', round(float(P[:,0].mean()),4))
```