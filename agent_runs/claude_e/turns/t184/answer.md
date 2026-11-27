The bin centres are stored per event (glucose_finger, glucose_lab, map_arterial, map_cuff). I'll write a value-head forward pass on the held-out head inputs WC: hidden layer, output layer, transition-table term, softmax averaged over draws, then the predicted mean.

I'll check it reproduces RR.pm with the original weights. Then I'll compare the shift from gated3(-1.9, 3) on drug-gated rows and ungated rows. A clear non-zero shift on ungated glucose rows would confirm that removing the repurposed units' original function is the extra cost.

```python
code2ev = {5:'map_cuff', 6:'glucose_finger', 7:'map_arterial', 10:'glucose_lab'}
CEN = {c: np.asarray(VOCAB['value_bin_centres'][e], float) for c,e in code2ev.items()}
print({c: len(v) for c,v in CEN.items()})
lvb = RR.lvb.values.astype(int)
def head_pm(Wd, Xd=WCs, codes=cs, lvb=lvb):
    W0_, b0_, W2_, TR_ = [ (Wd[k].numpy() if hasattr(Wd[k],'numpy') else np.asarray(Wd[k])).astype(np.float64) for k in ['W0','b0','W2','TR'] ]
    H = gelu(np.einsum('dnk,mk->dnm', Xd.astype(np.float64), W0_) + b0_)
    LG = np.einsum('dnm,bm->dnb', H, W2_) + b2 + TR_.reshape(-1,17)[codes*17 + lvb][None]
    LG = LG - LG.max(-1, keepdims=True); P = np.exp(LG); P /= P.sum(-1, keepdims=True); P = P.mean(0)
    pm = np.zeros(len(codes))
    for c, cv in CEN.items():
        m = codes==c
        cc = cv if len(cv)==17 else np.r_[np.nan, cv]
        pm[m] = (P[m][:,1:] * cc[1:]).sum(1) / P[m][:,1:].sum(1)
    return pm, P
pm0, P0 = head_pm(ORIG)
print('orig pm vs RR.pm: mean|diff| %.3f (z-draw noise expected ~0.5-0.7)' % np.abs(pm0 - RR.pm.values).mean())
pmE, PE = head_pm(Ecur)
sh = pmE - pm0
for nm, msk in [('gluc after ins', gl_rows & hi_s), ('gluc no ins', gl_rows & ~hi_s), ('ma after NE', (cs==7) & pv_ne), ('ma no NE', (cs==7) & ~pv_ne), ('mc', cs==5)]:
    print('%-15s n %5d  mean|shift| %.3f  mean shift %.3f' % (nm, msk.sum(), np.abs(sh[msk]).mean(), sh[msk].mean()))
```