`mystep` exposes everything I need: n3, the prelude FFN pre-activations and outputs, and h_new. edit_parts.pkl holds the decoder (`wxI`/`bxI`, `wxN`/`bxN`), the gate (`wg`/`bg`), the quiet directions q and vbar.

Next I'll trace, on the insulin and norepinephrine bundles of training stays, how each stage responds when the drug's rate is multiplied by 0.5, 0.75, 1.5 and 2:
1. the decoded residual x (read linearly from n3);
2. the gate s;
3. the GELU unit output (A − O);
4. the injected h_new along vbar.

If the linear part is fine, x should change by (f − 1) × rate with the same slope for f < 1 and f > 1. Whichever stage first shows a smaller slope for decreases is where the asymmetry seen in fwd_6 enters: stage 1 / u encoding if already in x, the gate if only in the unit outputs.

```python
import numpy as np, time
def internals(W, d):
    global WCUR
    old = WCUR; WCUR = W
    try:
        R = {}; mystep(d, R=R)
    finally: WCUR = old
    return R
ats_ = B_[3][1]
print('scale_atoms==ins_atoms_scaled:', str(scale_atoms(ats_,'insulin',2.0)) == str(ins_atoms_scaled(ats_,2.0)), scale_atoms(ats_,'insulin',0.5))
facs = [0.5, 0.75, 1.0, 1.5, 2.0]; rowsI = []; t0 = time.time()
for si, sid in enumerate(SIDS):
    Bs, Ss = bundles_B[sid], steps_all[sid]
    for j,(t,ats) in enumerate(Bs):
        k0 = [a for a in ats if a.get('kind')==0 and a['code'] in ('insulin','norepinephrine')]
        for drug in sorted(set(a['code'] for a in k0)):
            rate = [a['rate'] for a in k0 if a['code']==drug][0]
            rec = {'sid': sid, 'j': j, 'drug': drug, 'rate': rate, 'both': len(set(a['code'] for a in k0))==2}
            for f in facs:
                if f == 1.0: d2 = Ss[j]
                else:
                    e2,u2 = enc2(scale_atoms(ats, drug, f)); d2 = dict(Ss[j]); d2['u'] = u2
                R = internals(WE, d2); n3s = R['n3'][:4]; ph = R['ph'][:4]
                rec[f'xI{f}'] = float((n3s@EP_['wxI']+EP_['bxI']).mean()); rec[f'xN{f}'] = float((n3s@EP_['wxN']+EP_['bxN']).mean())
                rec[f's{f}'] = float((n3s@EP_['wg']+EP_['bg']).mean())
                rec[f'uI{f}'] = float((ph[:,502]-ph[:,140]).mean()); rec[f'uN{f}'] = float((ph[:,131]-ph[:,140]).mean())
                hn = R['h_new'].mean(0); rec[f'pI{f}'] = float(hn@vbar_I/(vbar_I@vbar_I)); rec[f'pN{f}'] = float(hn@vbar_N/(vbar_N@vbar_N))
            rowsI.append(rec)
    if time.time()-t0 > 700: print('time stop at stay', si); break
DF = pd.DataFrame(rowsI); print('rows', len(DF), DF.drug.value_counts().to_dict(), 'time %.0f' % (time.time()-t0))
for drug, X_ in [('insulin','I'), ('norepinephrine','N')]:
    g = DF[DF.drug==drug]
    for f in [0.5, 0.75, 1.5, 2.0]:
        dr = (f-1)*g.rate
        sl = lambda col: float(((g[f'{col}{f}']-g[f'{col}1.0'])*dr).sum()/(dr*dr).sum())
        print(drug, f, 'slope dx/drate %.3f  du/drate %.3f  dproj/drate %.3f | mean s %.3f  min s %.3f' % (sl('x'+X_), sl('u'+X_), sl('p'+X_), g[f's{f}'].mean(), g[f's{f}'].min()))
```