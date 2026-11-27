The bin tables complete what the head replica needs:
- Glucose finger bins run from below 78 to above 260 mg/dL.
- Insulin rate bins top out at 5 u/h, so doubled doses land in the top bin.
- Norepinephrine rate edges end in repeated 1.0s. Every doubled rate above the 1.0 cap therefore falls into bin 16, which norepinephrine never used in training. This is the out-of-range input I flagged earlier for the x2 edit.

Now I'll build the local value-head replica from the states, following ARCHITECTURE.md:
- view = out + out_norm(decayed slot mean) + gap(dt) + z_head(prior mean z);
- query norm, then code embedding plus the "last value" register terms;
- the two-layer value network plus the transition table.

I'll scan the unknown details (activation, type row, register layout, status row, gap scaling) for the variant that best reproduces the tool's per-reading predicted means.
- A close match, with MAE well under 1 mg/dL, validates the replica for designing the edit.
- A poor match means some piece of the head is misunderstood.

```python
import numpy as np, torch
Wt=lambda n: load_original(n).numpy().astype(np.float64)
def LNf(x,p,eps=1e-5):
    m=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-m)/np.sqrt(v+eps)*Wt(p+'.weight')+Wt(p+'.bias')
from scipy.special import erf
ACTS={'relu':lambda x:np.maximum(x,0),'gelu':lambda x:0.5*x*(1+erf(x/np.sqrt(2)))}
def L(x,p,bias=True): 
    y=x@Wt(p+'.weight').T
    return y+Wt(p+'.bias') if bias else y
def M2(x,p,act): return L(ACTS[act](L(x,p+'.0')),p+'.2')
CODE={'map_cuff':5,'glucose_finger':6,'map_arterial':7,'glucose_lab':10}
fa=fa2.copy(); fa['code']=fa.event.map(CODE)
# previous state row
order=np.lexsort((hr_,sid_)); assert (order==np.arange(len(order))).all()
starts={s:np.where(sid_==s)[0] for s in np.unique(sid_)}
prow=[]
for s_,t in zip(fa.stay_id,fa.hour):
    ii=starts[s_]; j=ii[hr_[ii]<t-1e-6]; prow.append(j[-1] if len(j) else -1)
fa['prow']=prow; fa=fa[fa.prow>=0].reset_index(drop=True)
# register: last earlier value of same code
lastv=[];lastage=[]
for s_,t,ev in zip(fa.stay_id,fa.hour,fa.event):
    ser=Ptu.loc[s_,ev]; ser=ser[(ser.index<t)&ser.notna()]
    if len(ser): lastv.append(ser.iloc[-1]); lastage.append((t-ser.index[-1])*3600)
    else: lastv.append(np.nan); lastage.append(np.nan)
fa['lastv']=lastv; fa['lastage']=lastage
def zsc(ev,x): e=num[(ev,'value')]; return np.clip((x-e['median'])/e['iqr'],-10,10)
def fbin(ev,x): e=num[(ev,'value')]; return 1+np.searchsorted(np.array(e['fine_bin_edges']),x,side='left')
fa['has']=fa.lastv.notna().astype(float)
fa['zl']=[zsc(ev,v) if v==v else 0.0 for ev,v in zip(fa.event,fa.lastv)]
fa['lb']=[int(fbin(ev,v)) if v==v else 0 for ev,v in zip(fa.event,fa.lastv)]
fa['la']=np.where(fa.has>0,np.log1p(fa.lastage.fillna(0))/10,0.0)
Xp=X_[fa.prow.values]; outp=Xp[:,:192]; hp=Xp[:,192:960].reshape(-1,4,192)
dt=(fa.hour.values-hr_[fa.prow.values])*3600
dec=np.exp(-dt[:,None]*np.exp(Wt('core.flow.a'))[None,:])
hm=LNf((hp*dec[:,None,:]).mean(1),'core.out_norm')
cent={k:np.array(v) for k,v in vc.items()}
CM=np.stack([cent[ev] for ev in fa.event])
obs=fa.predicted_mean.values
TT=Wt('embed_transition.value_bin.weight'); TE=Wt('type_emb.weight'); CE=Wt('embed.code.weight')
VB=Wt('embed.bins.value.weight'); RB_=Wt('embed.bins.rate.weight'); LS=Wt('last_status.weight')
res=[]
for act in ['relu','gelu']:
  for gs in [10.0,1.0]:
    gp=M2(np.log(dt)[:,None]/gs,'gap',act)
    ctx=outp+hm+gp
    zpm=M2(ctx,'z_prior',act)[:,:32]
    view=ctx+L(zpm,'z_head')
    for lnv in ['A','B']:
        f8=np.zeros((len(fa),8))
        if lnv=='A': f8[:,0]=fa.has; f8[:,1]=fa.la; f8[:,2]=fa.zl
        else: f8[:,0]=fa.has; f8[:,1]=fa.la; f8[:,2]=fa.zl; f8[:,3]=fa.has
        lnum=L(f8,'last_numeric')
        for st in [None,0,1]:
          base=CE[fa.code.values]+lnum+VB[fa.lb.values]+RB_[0]+(0 if st is None else LS[st])
          for ty in range(15):
            q=LNf(view+TE[ty],'query_norm')
            lg=M2(q+base,'attr_heads.value_bin',act)+TT[fa.code.values*17+fa.lb.values]
            lg=lg-lg.max(1,keepdims=True); p=np.exp(lg); p/=p.sum(1,keepdims=True)
            pm=(p[:,1:]*CM).sum(1)
            res.append((np.abs(pm-obs).mean(),act,gs,lnv,st,ty))
res.sort(key=lambda r:r[0]); print(len(fa)); print(res[:6])
```