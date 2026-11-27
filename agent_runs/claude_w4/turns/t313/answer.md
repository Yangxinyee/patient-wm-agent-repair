Run 277 output, in short:

- **Saved edit.** It is the v4 family:
  - prelude head-1 value rows (kv), large norm
  - head-1 output columns
  - b_in
- **fwd_9.** A factual run on 100 tuning stays with some edit; I have not yet read its NLL.
- **v5_parts.** I also built a "v5" (new b_in targets plus per-reading readouts) in the blocked runs, with no notes.

**New hypothesis for v4's failure.** I re-read the fwd_8 pattern: x2 raised glucose, x1/2 lowered it strongly, and rollouts drifted up even without insulin. Two candidate causes fit this pattern.

1. **The insulin "presence" readout depends on the rate.** The edit subtracts 1.77·presence to centre the injection. If doubling the rate inflates presence and halving deflates it, this term outweighs the rate term. That would explain both the wrong sign and the asymmetry.
2. **The channel direction has natural activity at measurement bundles.** The b_in edit replaced the model's own response there, which would produce drift.

**This run checks, printing aggregates only:**

- fwd_9's NLL change, and which targets (v4 or v5) and gain are in the saved b_in;
- the three head-1 readouts (rate, NE, presence) on recorded drug tokens, measurement tokens and the no-action token;
- the channel's natural activity at measurement bundles, with original versus edited weights;
- which helper functions exist, for example one that embeds an arbitrary dose.

**How to read the results:**

- **Confirms cause 2:** natural channel activity is about 0.01 with original weights → the drift is natural activity.
- **Confirms a head-1 leak:** activity appears only with edited weights → the readouts must be made to vanish on measurement tokens.

```python
import json, numpy as np, torch
R=json.load(open('/work/fwd_9/summary.json'))['results']
def br(d):
    if isinstance(d,dict): return {k:br(v) for k,v in list(d.items())[:8]}
    if isinstance(d,float): return round(d,4)
    if isinstance(d,list): return f'list{len(d)}'
    return d
print('fwd9:',str(br(R))[:500])
V4=np.load('/work/v4_parts.npz'); V5=np.load('/work/v5_parts.npz')
pnw=load_original('core.prelude_norm.weight').numpy(); pnb=load_original('core.prelude_norm.bias').numpy()
Wb=load('core.b_in.weight').numpy()
for nm in ['I','N']:
    q=V4['q'+nm]; r=Wb@(pnw*q)
    for tn,t in [('v4',V4['t'+nm]),('v5',V5['t'+nm+'5'])]:
        print(nm,tn,'cos %.4f scale %.4f'%(r@t/np.linalg.norm(r)/np.linalg.norm(t), r@t/(t@t)))
print('qs==v4 q:',float(np.abs(qs-np.stack([V4['qI'],V4['qN']],1)).max()), 'gp==pnw:',float(np.abs(gp-pnw).max()), 'v4 rbar,c:',float(V4['rbar']),float(V4['c']))
fn=[k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__']
print('funcs:',str(fn)[:700])
def ln(x,w,b):
    m=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-m)/np.sqrt(v+1e-5)*w+b
kv0=load_original('core.prelude.0.attn.kv.weight').numpy(); kvc=load('core.prelude.0.attn.kv.weight').numpy()
n1w=load_original('core.prelude.0.norm1.weight').numpy(); n1b=load_original('core.prelude.0.norm1.bias').numpy()
D=(kvc-kv0)[240:288]; Mi=np.linalg.inv(M)
def fr(U): return (Mi@(A.T@(D@ln(np.asarray(U,dtype=np.float64),n1w,n1b).T))).T
Z=np.load('/work/rep_drug.npz'); u=Z['u']; rI=Z['rI']; rN=Z['rN']
fd=fr(u); tgt=np.stack([rI,(rN-.52)*(rN>0),(rI>0)*1.],1)
print('drug readout rms err',np.sqrt(((fd-tgt)**2).mean(0)).round(4),' frac rI>0 %.2f rN>0 %.2f'%((rI>0).mean(),(rN>0).mean()))
print('f(u_const)',fr(Z['u_const'][None])[0].round(4))
for k in range(6): print('Tm tok',k,'mean f',fr(Tm[:,k]).mean(0).round(4),'sd',fr(Tm[:,k]).std(0).round(4), 'dist u_const %.2f'%np.linalg.norm(Tm[:,k]-Z['u_const'],axis=1).mean())
Wo0=load_original('core.prelude.0.attn.out.weight').numpy(); Woc=load('core.prelude.0.attn.out.weight').numpy()
def chs(T,Wkv,Wout):
    x1,x2=prelude_b(T,Wkv,Wout); x2=np.asarray(x2,dtype=np.float64); E=ln(x2,pnw,pnb)
    return np.stack([E@(V4['qI']/pnw),E@(V4['qN']/pnw)],-1)
try:
    s0=chs(Tm,kv0,Wo0); s1=chs(Tm,kvc,Woc)
    print('meas chan orig mean',s0.mean((0,1)).round(5),'sd',s0.std((0,1)).round(5),'| edited mean',s1.mean((0,1)).round(5),'sd',s1.std((0,1)).round(5))
except Exception as e: print('prelude_b err',repr(e)[:200])
```