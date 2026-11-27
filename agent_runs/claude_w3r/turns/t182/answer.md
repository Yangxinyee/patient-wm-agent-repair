A linear readout of the normalised u token, over factual rates plus rates ×2 and ×½, has held-out residual sd 0.091 for insulin and 0.051 for norepinephrine (max about 0.6). That is usable but not perfect.

What matters most is different for each use. On factual rates the residual sd drives the factual cost. On edited rates any mean bias directly scales the readouts. Next I'll split the held-out residuals by edit type (factual, insulin ×2/×½, norepinephrine ×2/×½) and report the mean bias and the slope of predicted on true rate. Small bias (≲ 3%) everywhere would confirm the linear channel; a large bias would mean I need to reweight the fit.

```python
ET = []
for s in sorted(recs.stay_id.unique()):
    Bs, rs, EUs = stay_data(int(s))
    for k in range(1, len(Bs)):
        at = Bs[k][1]; hasI = any(a['code']==ci and a['kind']==0 for a in at); hasN = any(a['code']==cn and a['kind']==0 for a in at)
        if not (hasI or hasN): continue
        ET.append('fact')
        if hasI: ET += ['ins2', 'insh']
        if hasN: ET += ['ne2', 'neh']
ET = np.array(ET); print(len(ET), len(UX))
A_i, A_n = fit_lin(UXh, YXi, tr_), fit_lin(UXh, YXn, tr_)
for nm, Y, A_ in [('ins', YXi, A_i), ('ne', YXn, A_n)]:
    pr = UXh @ A_
    for et in ['fact','ins2','insh','ne2','neh']:
        m_ = (ET==et) & ~tr_ & (Y>0)
        if m_.sum()==0: continue
        sl = np.polyfit(Y[m_], pr[m_], 1)[0]
        print(nm, et, 'n', m_.sum(), 'bias', (pr[m_]-Y[m_]).mean().round(4), 'resid sd', (pr[m_]-Y[m_]).std().round(4), 'slope', round(sl,3), 'mean Y', Y[m_].mean().round(3))
```