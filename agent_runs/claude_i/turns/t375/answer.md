The 2×QD dose_edit settles the sign question.

| Readout | 2×QD | Original | Change |
|---|---|---|---|
| Insulin x2 | +0.15 | −2.01 | +2.16 |
| Insulin x1/2 | +0.18 | +1.16 | −0.98 |
| NE x2 | +0.05 | +0.37 | −0.32 |
| NE x1/2 | +0.05 | −0.06 | +0.12 |

For both drugs the changes from the original have opposite signs for x2 and x1/2, at roughly the −1/2 ratio that a linear channel would give. So the clean channel works linearly, but with the opposite sign to my design. That sign had been checked only on the leaky version, which also explains why DW2 damped the effects.

This also fits the factual results. With the designed sign, glucose_finger NLL got worse, as it should if that direction opposes the true drug effect. With the sign flipped, the channel should push all four readouts toward the targets, and the factual fit may get better rather than worse.

I have 3 Python runs left, so this run picks the flipped gains directly instead of testing one guess:
1. Split the quiet delta into its insulin (z18) and NE (z28) parts so each gets its own gain.
2. From the per-reading fwd_14 results, estimate how factual NLL varies with each gain (for each output, a linear mean-shift term plus a quadratic term fitted to the k=2 result).
3. Pick the most negative gains, capped at −5, whose predicted NLL cost is at most 0.0025. This assumes the readouts change linearly with gain.
4. Install the chosen delta.

The next factual and dose_edit passes check it:
- **Confirmed:** factual within about +0.003, insulin x2 clearly below −2.0, and NE x2 above 0.37.
- **Refuted:** factual fails or the shifts do not move as predicted. Then run 299 reduces or reverts.

```python
import numpy as np, pandas as pd, torch
def A(x):
    try: return x.detach().cpu().numpy()
    except AttributeError: return np.asarray(x)
def SE(n,x):
    x=np.ascontiguousarray(np.asarray(x,dtype=np.float32))
    try: save_edit(n,x)
    except Exception: save_edit(n,torch.from_numpy(x))
kvn='core.prelude.0.attn.kv.weight'
Mi_=np.linalg.inv(Msum+1e-3*tr*np.eye(192)); Km=np.linalg.solve(Cm.T@Mi_@Cm, Cm.T@Mi_)
RI=R.copy(); RI[:,7:]=0; RN=R.copy(); RN[:,4:7]=0; DI=RI@Km; DN=RN@Km
QD=np.load('/work/QD.npy'); print('chk %.2e'%(np.linalg.norm(DI+DN-QD)/np.linalg.norm(QD)))
kI,kN=-2.0,-2.0
try:
    d='/work/fwd_14'; o=pd.read_csv(f'{d}/factual_original.csv'); e=pd.read_csv(f'{d}/factual_edited.csv')
    cols=list(o.columns); print(cols)
    ev=[c for c in cols if 'event' in c.lower()][0]; nl=[c for c in cols if 'nll' in c.lower()][0]; mc_=[c for c in cols if 'mean' in c.lower()][0]
    yc=[c for c in cols if c.lower() in ('value','observed','y','reading','obs')] or [c for c in cols if 'value' in c.lower() and 'mean' not in c.lower() and 'nll' not in c.lower()]
    yc=yc[0]; N=len(o); W={}; Aa={}; Bb={}
    for g in o[ev].unique():
        mk=(o[ev]==g).values; r=(o[yc]-o[mc_]).values[mk]; dm=(e[mc_]-o[mc_]).values[mk]/2.0
        s2=np.mean(r**2); a=float(np.mean(-dm*r/s2)); act=float(np.mean((e[nl]-o[nl]).values[mk]))
        b=(act-2*a)/4
        if b<0: b=abs(act)/4; a=(act-4*b)/2
        W[g]=mk.sum()/N; Aa[g]=a; Bb[g]=b; print(g, 'n %d a %.5f b %.5f act2 %.4f'%(mk.sum(),a,b,act))
    G=[g for g in W if 'gluc' in g]; Mp=[g for g in W if 'map' in g]
    def nll(ki,kn): return sum(W[g]*(Aa[g]*ki+Bb[g]*ki*ki) for g in G)+sum(W[g]*(Aa[g]*kn+Bb[g]*kn*kn) for g in Mp)
    def obj(ki,kn):
        x2=-2.007+1.08*ki; hf=1.164-0.4925*ki; n2=0.367-0.1575*kn; nh=-0.064+0.0575*kn
        return 2*abs(x2+12.5)/12.5+abs(hf-6.25)/6.25+abs(n2-4.33)/4.33+abs(nh+2.17)/2.17
    best=None
    for ki in np.arange(-5,0.01,0.25):
        for kn in np.arange(-5,0.01,0.25):
            p=nll(ki,kn)
            if p<=0.0025 and (best is None or obj(ki,kn)<best[0]): best=(round(obj(ki,kn),3),float(ki),float(kn),round(p,4))
    print('best', best, 'pred(-2,-2) %.4f pred(2,2) %.4f'%(nll(-2,-2),nll(2,2)))
    if best is not None: kI,kN=best[1],best[2]
except Exception as ex: print('ERR', repr(ex)[:200])
D=kI*DI+kN*DN; SE(kvn, A(load_original(kvn)).astype(np.float64)+D); np.savez('/work/QDparts.npz',DI=DI,DN=DN)
print('INSTALLED kI %.2f kN %.2f'%(kI,kN), edited())
open('/work/notes.md','a').write('\n- fwd14 factual 2xQD +0.0043 (gf +0.011). fwd15 dose 2xQD: ins x2 +0.15, half +0.18, ne x2 0.05, half 0.05 -> clean channel is LINEAR with WRONG sign (changes vs orig: +2.16,-0.98,-0.32,+0.12).\n- run298: split QD into DI (z18) and DN (z28) (/work/QDparts.npz); installed kv=orig+kI*DI+kN*DN with NEGATIVE gains chosen by factual model (pred <=0.0025). kI=%.2f kN=%.2f. Runs left: 299, 300 (keep 300 for revert).\n'%(kI,kN))
```