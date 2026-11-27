Making the readouts leak-free costs accuracy, and the cost is steep:

- **Unconstrained.** The readouts are accurate: rate rms error 0.11 units/h for insulin and 0.06 mcg/kg/min for NE on recorded doses. But the attention-weighted leak reaches 5,087 units at a rare bundle and 2 units at a non-drug bundle.
- **Constrained.** Even mild constraints raise the recorded-dose error to 0.4–0.7 for insulin. Worse, the readout of the no-action token drifts to +0.8 to +9. That token is what head 1 reads at every measurement bundle, so this alone would inject a fake dose everywhere.

Leakage only bites where head 1 actually attends to the other tokens, which is rare: 7 of 2,319 drug bundles exceed 5%, and none of the sampled non-drug bundles do. So a mildly constrained readout with the no-action token pinned hard to zero may be good enough. A leak of a few units at 0.3% of bundles costs well under 0.001 nats.

Next I'll refit with the no-action token at weight 10⁴ and the other-token penalty at 10⁻³ to 10⁻², plus a variant that also penalises the actual attention-weighted leak at the sampled bundles. I'll evaluate each on:
- recorded doses;
- the multiplied-dose tokens, if the cached dicts `N3`/`VAR` hold them (I'll check their contents first);
- the no-action readout;
- the leak.

```python
for nmD in ['N3','VAR','resI']:
    D_=globals()[nmD]; k0=list(D_.keys())[0]; v0=D_[k0]
    print(nmD, 'keys', list(D_.keys()), '| value type', type(v0).__name__, getattr(v0,'shape',None) if not isinstance(v0,(list,tuple,dict)) else (len(v0), type(v0[0]).__name__ if isinstance(v0,(list,tuple)) else list(v0.keys())[:5]))
Ln=LN1(u_none)
Lb_d=np.einsum('nqj,njd->nqd',A_d[:,:,[0,1,2,3,4,6]],LN1(np.asarray(TOK)[:,[0,1,2,3,4,6]].reshape(-1,192)).reshape(len(TOK),6,192)).reshape(-1,192)
Lb_n=np.einsum('nqj,njd->nqd',A_n[:,:,[0,1,2,3,4,6]],LN1(np.asarray(TOKn)[:,[0,1,2,3,4,6]].reshape(-1,192)).reshape(len(TOKn),6,192)).reshape(-1,192)
Lb=np.concatenate([Lb_d,Lb_n])
cands={}
for nm,y_,c_,pm_ in [('ins',yIr_c,cI,pIu),('ne',yNr,cN,pNu)]:
    yc=np.where(pm_,y_-c_,0.0)
    for wO,wL in [(1e-3,0),(3e-3,0),(1e-2,0),(1e-3,1e3),(3e-4,1e4)]:
        Aq=L_U.T@L_U+wO*(L_O.T@L_O)+wL*(Lb.T@Lb)+1e4*np.outer(Ln[0],Ln[0])+1e-6*np.eye(192); r=np.linalg.solve(Aq,L_U.T@yc)
        e=L_U@r-yc; ld=leak(r,TOK,A_d); ln_=leak(r,TOKn,A_n)
        print(f'{nm} wO={wO:g} wL={wL:g}: |r| {np.linalg.norm(r):8.1f} rms all {np.sqrt((e**2).mean()):.3f} recorded {np.sqrt((e[len(UG)+len(UR):-1]**2).mean()):.3f} none {(Ln@r).item():+.5f} | other rms {np.sqrt(((L_O@r)**2).mean()):.2f} | leak drug med {np.median(ld):.5f} p99 {np.percentile(ld,99):.4f} max {ld.max():.3f}; non-drug p99 {np.percentile(ln_,99):.4f} max {ln_.max():.4f}')
        cands[(nm,wO,wL)]=r
```