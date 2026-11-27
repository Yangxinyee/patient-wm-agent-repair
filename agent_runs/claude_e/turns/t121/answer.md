`make_edit` builds the edit from the simpler readouts `RO_i` and `RO_n`, plus globals `units`, `gvec`, `mid`, `tg` and `tm`. `fit_readout` regresses the dose target on the head input together with last-bin dummies for each code. So each readout's `b` (2×17) is the expected-dose offset per (code, last bin), not a logit direction. In `make_edit` that last-bin part goes into the transition table, which is why the table rows for codes 5, 6, 7 and 10 change.

That fits my design: the correction acts on the dose minus what the current glucose or MAP level implies. In a rollout, the subtracted part follows the model's own sampled readings.

Next I will look at `mid`, `tg`, `tm` and `gvec`. I will call `make_edit` with unit coefficients and compare it with the stored ED, to find which (a_ins, a_ne) ED was built with. If ED's changed columns are a multiple of the unit-coefficient edit, that multiple is its coefficient and the edit is linear in a. Otherwise make_edit has extra structure that I must understand before trusting it.

```python
for nm in ['mid','tg','tm','gvec','CENT','obin']:
    if nm in globals():
        v=globals()[nm]; print(nm, type(v).__name__, (np.round(np.asarray(v,dtype=float),3) if np.size(v)<=40 else ('shape',np.shape(v),'norm',float(np.linalg.norm(np.asarray(v,dtype=float))))))
    else: print(nm,'MISSING')
E1=make_edit(1.0,1.0); E2=make_edit(2.0,2.0)
for k in ['W0','b0','W2','TR']:
    d1=(E1[k]-ORIG[k]); d2=(E2[k]-ORIG[k]); de=(ED[k]-ORIG[k])
    print(k,'|E1-O|',float(d1.abs().sum()),'|E2-O|',float(d2.abs().sum()),'|ED-O|',float(de.abs().sum()))
for j,u in enumerate(units):
    for k in ['W2']:
        a1=(E1[k][:,u]-ORIG[k][:,u]); ae=(ED[k][:,u]-ORIG[k][:,u]); a2=(E2[k][:,u]-ORIG[k][:,u])
        r=float((ae*a1).sum()/(a1*a1).sum()); r2_=float((a2*a1).sum()/(a1*a1).sum())
        print('unit',u,'W2 col: ED/E1 ratio %.3f  E2/E1 %.3f  resid %.4f'%(r,r2_,float((ae-r*a1).abs().max())))
    print('   W0 row diff ED vs E1 max', float((ED['W0'][u]-E1['W0'][u]).abs().max()), ' b0', float(ED['b0'][u]), float(E1['b0'][u]), float(ORIG['b0'][u]))
```