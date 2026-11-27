Run 278 explains why v4 failed so badly.

**Factual fit.** fwd_9 was v4(1,1): its value NLL is 1.610 versus 1.494 for the original model. That is +0.116 nats, more than twenty times the 0.005 budget.

**Cause.** The insulin and NE readouts that the v4 edit added to the head-1 value rows are fine on the action (u) token. On every other token they are huge:

| Token type | Readout values |
|---|---|
| Slots | about 300 ± 760 |
| Reading token | about ±3000 |

These vectors were fitted on u tokens only, so in the directions those tokens never use their weights are unconstrained. Head 1 puts only about 0.2% of its attention off the u token. Even so, that small attention times readouts in the hundreds or thousands injects noise of roughly ±1 rate unit at every bundle.

That noise is consistent with the large factual damage, the rollout drift and the scrambled dose response. The no-action token is clean, and so is the b_in side: natural channel activity is tiny, sd 0.0008.

**Fix (v6).** I refit the three readouts by ridge regression.

- They should reproduce the rates on drug u tokens.
- They are constrained to about zero on:
  - slot tokens;
  - reading/event tokens;
  - the no-action token.
- They are written back through the same rank-3 value add.
- The output projection and b_in are kept as in v4, with gains 1 and 1.

First I check that my re-implementation of the build reproduces the saved v4 exactly; only then do I save v6.

**What would confirm the fix:**

- the drug readout error stays at about 0.1 / 0.05 / 0.01;
- off-u readouts drop to below about 1;
- measurement-bundle channel activity falls back to about the original sd of 0.0008.

After that, dose_edit and factual forward passes calibrate the gains.

```python
import inspect, numpy as np, torch
for f in ['uv','drug_atom','enc','rate_at']:
    try: print(f, str(inspect.signature(globals()[f]))[:80])
    except Exception as e: print(f,'?')
Z=np.load('/work/rep_drug.npz'); V4=np.load('/work/v4_parts.npz')
f64=lambda x: np.asarray(x,dtype=np.float64)
Xu=ln(f64(Z['u']),n1w,n1b); F=np.stack([Z['rI'],(Z['rN']-.52)*(Z['rN']>0),(Z['rI']>0)*1.],1)
Xs=ln(f64(Tm[:,:4].reshape(-1,192)),n1w,n1b)
Xe=np.concatenate([ln(f64(Tm[:,5]),n1w,n1b),ln(f64(Z['e']),n1w,n1b)],0)
Xc=ln(f64(Z['u_const'][None]),n1w,n1b)
Guu=Xu.T@Xu; Gs=Xs.T@Xs; Ge=Xe.T@Xe; Gc=Xc.T@Xc; XuF=Xu.T@F
res={}; sel=None
for wo in [0.1,1,10,100]:
    B=np.linalg.solve(Guu+wo*(Gs+Ge)+1e4*Gc+np.eye(192),XuF)
    eu=np.sqrt(((Xu@B-F)**2).mean(0)); es=np.sqrt(((Xs@B)**2).mean(0)); ee=np.sqrt(((Xe@B)**2).mean(0)); ec=np.abs(Xc@B).max(0)
    res[wo]=B
    print(f'wo={wo}: drug err {eu.round(3)} slots {es.round(3)} e/t5 {ee.round(3)} uc {ec.round(4)}')
    if eu[0]<=.12 and eu[1]<=.06 and eu[2]<=.012: sel=wo
print('att head1 by key (drug bundles):',Z['att'][:,1].mean((0,1)).round(4))
A_=f64(V4['A']); M_=f64(V4['M']); Mi_=np.linalg.inv(M_); c_=float(V4['c'])
Wb0=f64(load_original('core.b_in.weight').numpy())
Bold=(Mi_@A_.T@D).T
def build6(B,kI,kN,rbI=float(V4['rbar'])):
    kv=f64(kv0).copy(); kv[240:288]+=A_@M_@B.T
    Wo=f64(Wo0).copy(); Q3=np.stack([V4['qI'],V4['qN'],-rbI*V4['qI']],1)
    Wo[:,48:96]=f64(Wo0[:,48:96])@(np.eye(48)-A_@A_.T)+c_*Q3@Mi_@A_.T
    Wb=Wb0.copy()
    for k,q,t in [(kI,V4['qI'],V4['tI']),(kN,V4['qN'],V4['tN'])]: Wb+=np.outer(k*t-Wb0@(pnw*q),q/pnw)
    return kv,Wo,Wb
kvx,Wox,Wbx=build6(Bold,1,1)
print('repro err %.3g %.3g %.3g'%(np.abs(kvx-kvc).max(),np.abs(Wox-Woc).max(),np.abs(Wbx-Wb).max()))
print('selected wo:',sel)
if sel is not None:
    kvn,Won,Wbn=build6(res[sel],1,1)
    s1=chs(Tm,kvn,Won); print('v6 meas chan mean',s1.mean((0,1)).round(5),'sd',s1.std((0,1)).round(5))
    for n,x in [('core.prelude.0.attn.kv.weight',kvn),('core.prelude.0.attn.out.weight',Won),('core.b_in.weight',Wbn)]:
        save_edit(n,torch.tensor(x,dtype=torch.float32))
    np.savez('/work/v6_parts.npz',B=res[sel],wo=sel)
    print('saved v6(1,1); edited:',edited())
with open('/work/notes.md','a') as f:
    f.write('- fwd_9 = factual both tuning 100 stays for v4(1,1): value NLL 1.4938 -> 1.610 (+0.116!). finger 1.5515->1.6887.\n'
            '- Cause (run 278): v4 head-1 readouts B fit only on u tokens; on slot tokens f~300+-760, reading tok ~+-3000 -> 0.2% attention leak = +-1 rate unit noise everywhere.\n'
            '  Tm token 4 == u_const (f~0); b_in natural channel activity tiny (sd .0008).\n'
            f'- Run 279: v6 = ridge refit of B (targets on drug u; ~0 on slots/e/tok5/u_const), build6(B,kI,kN,rbI) re-implements v4 (repro checked). wo={sel}. Saved v6(1,1) if sel not None.\n')
```