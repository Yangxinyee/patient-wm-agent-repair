The states for 20 stays are in. This run looks at four things:
- **z dims 18/28 by position in the stay.** Their values at the first, second and third bundle, and how far the edited state drifts from the original after each, show whether the misfire starts at the first bundle and whether it comes through the z_post channel.
- **The same channel later in the stay.** Its size at insulin, norepinephrine and drug-free bundles, to check that the gating works there.
- **Slot tokens through the attention edit.** How strongly the first bundle's slot tokens (zero state, so just the slot embeddings) excite the edited attention weights, compared with normal slot tokens.
- **A first-bundle detector.** Whether the output layer-norm gives an exact linear signature of a zero previous state, which would let me switch the channel off at the first bundle.

How to read the result:
- Large z18/z28 at bundle 0 means the z_post units misfire.
- Small z but a large state change, plus a large slot-token response, means the attention edit leaks through the slot tokens.

```python
import numpy as np
S0=np.load('/work/fwd_7/states_original.npz'); S1=np.load('/work/fwd_7/states_edited.npz')
k2=[k for k in S0.files if S0[k].ndim==2][0]; x0=S0[k2].astype(np.float64); x1=S1[k2].astype(np.float64)
sid=S0['stay_id']; hr=S0['hour']; hi=S0['has_insulin'].astype(bool); hn=S0['has_norepinephrine'].astype(bool)
bi=np.zeros(len(sid),int)
for s in np.unique(sid):
    w=np.where(sid==s)[0]; bi[w]=np.arange(len(w))
rel=np.linalg.norm(x1[:,:960]-x0[:,:960],axis=1)/np.linalg.norm(x0[:,:960],axis=1)
z18=x1[:,978]; z28=x1[:,988]; o18=x0[:,978]; o28=x0[:,988]
oth=[i for i in range(32) if i not in (18,28)]
dzo=np.abs(x1[:,960:992][:,oth]-x0[:,960:992][:,oth]).max(1)
for b in range(5):
    m=bi==b
    print(b,'hr',np.unique(hr[m])[:3],'rel',rel[m].mean().round(3),'z18',z18[m].mean().round(2),z18[m].std().round(2),'z28',z28[m].mean().round(2),z28[m].std().round(2),'o18,o28',o18[m].mean().round(3),o28[m].mean().round(3),'dzoth',dzo[m].max().round(3),'ins',hi[m].mean().round(2),'ne',hn[m].mean().round(2))
m=bi>=5
for nm_,g in [('ins',hi&m),('ne',hn&m),('none',~hi&~hn&m)]:
    print(nm_,g.sum(),'|z18|',np.abs(z18[g]).mean().round(3),np.abs(z18[g]).max().round(2),'|z28|',np.abs(z28[g]).mean().round(3),np.abs(z28[g]).max().round(2),'rel',rel[g].mean().round(3))
cand=[n for n in NAMES if any(k in n for k in ['slot','gap','out_norm','flow'])]+[n for n in NAMES if n.startswith('core.prelude.0') and 'attn' not in n]
print([(n,tuple(SHAPES[n])) for n in cand][:24])
def LN(v,g,b): m=v.mean(-1,keepdims=True); va=v.var(-1,keepdims=True); return (v-m)/np.sqrt(va+1e-5)*g+b
DW=np.load('/work/DW_kv_full.npy').astype(np.float64)[192:]
try:
    sn=[n for n in NAMES if 'slot' in n and np.prod(SHAPES[n])%192==0][0]; sl=np.asarray(load(sn),dtype=np.float64).reshape(-1,192)
    pn=[n for n in NAMES if n.startswith('core.prelude.0') and 'attn' not in n and n.endswith('weight') and tuple(SHAPES[n])==(192,)]
    g=np.asarray(load(pn[0]),dtype=np.float64); b=np.asarray(load(pn[0][:-6]+'bias'),dtype=np.float64)
    inj_first=np.linalg.norm(DW@LN(sl,g,b).T,axis=0)
    prev=np.where(bi>=1)[0]-1; injn=[]
    for j in range(sl.shape[0]):
        tok=x0[prev,192+192*j:384+192*j]+sl[j]; injn.append(np.linalg.norm(DW@LN(tok,g,b).T,axis=0))
    injn=np.array(injn); print('slot',sn,'LN',pn[0],'inj first',inj_first.round(2),'normal med',np.median(injn,1).round(3),'p99',np.percentile(injn,99,1).round(3))
except Exception as ex: print('EK',ex)
try:
    on=[n for n in NAMES if 'out_norm' in n and n.endswith('weight')][0]
    go=np.asarray(load(on),dtype=np.float64); bo=np.asarray(load(on[:-6]+'bias'),dtype=np.float64)
    r=1/go; K0=(bo*r).sum(); ro=x0[:,:192]@r
    print('out_norm',on,'K0',round(K0,3),'r.out mean',ro.mean().round(3),'sd',ro.std().round(5),'min|go|',np.abs(go).min().round(4))
except Exception as ex: print('EO',ex)
```