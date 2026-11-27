The run failed on a numpy detail: `float()` on a one-element array. Before it failed, it showed two useful things.

- **Head-1 attention to the slots exceeds 5% at only 7 of 2,319 drug bundles and none of the 520 sampled non-drug bundles.** So leakage only needs to be controlled tightly in rare cases.
- **The action-token fit set covers edited doses up to 22.5 units/h insulin and 3.0 mcg/kg/min NE.** The mean recorded rates when present are 1.77 and 0.54. Earlier I found the input z-score clips at 10, so insulin rates above 11.5 are indistinguishable to the model.

Next I'll rerun the same fit with the scalar fixed and insulin targets capped at 11.5, matching what the model can actually see. Per drug and per constraint weight, I'll report readout accuracy, the no-action readout, and the attention-weighted leak.
- **Usable readout found:** some weight gives rate error below about 0.1 and a maximum leak below about 0.02 units.
- **Trade-off too steep:** I'd have to sharpen head-1 attention instead.

```python
yIr_c=np.minimum(yIr,11.5)
for nm,y_,c_,pm_ in [('ins',yIr_c,cI,pIu),('ne',yNr,cN,pNu)]:
    yc=np.where(pm_,y_-c_,0.0)
    for wO in [0.,1e-2,1e-1,1.,10.]:
        Aq=L_U.T@L_U+wO*(L_O.T@L_O)+1e-6*np.eye(192); r=np.linalg.solve(Aq,L_U.T@yc)
        e=L_U@r-yc; ld=leak(r,TOK,A_d); ln_=leak(r,TOKn,A_n)
        print(f'{nm} wO={wO:6g}: |r| {np.linalg.norm(r):9.1f} rms err all {np.sqrt((e**2).mean()):.4f} (recorded {np.sqrt((e[len(UG)+len(UR):-1]**2).mean()):.4f}) none {(LN1(u_none)@r).item():+.4f} | other |r.x| rms {np.sqrt(((L_O@r)**2).mean()):.3f} | leak drug med {np.median(ld):.4f} p99 {np.percentile(ld,99):.3f} max {ld.max():.3f}; non-drug max {ln_.max():.3f}')
        globals()[f'r_{nm}_{wO:g}']=r
```