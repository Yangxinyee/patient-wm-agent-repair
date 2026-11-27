`step_at` returns (new h as 4×192, out as 192). Note that `a_y` is not a quiet y-direction (natural sd 0.77; the quietest y-directions have sd ≈0.004). What matters is the head-side read noise, though, and `a_y` was chosen earlier because its with_code response is consistent.

Now I compare, for this stay, the edited minus original h (slot mean) projected on the two target y-directions against the decoded rates from my numeric pipeline. Confirmation would be the projection ≈10 × decoded rate at action bundles, ≈0 elsewhere, with the off-target part small.

```python
nb=len(Bq); dec_b={}
for nm,dd in DECL.items(): dec_b[nm]=((Hf[:,dd['units'][0]]-Hf[:,dd['units'][1]])/(2*dd['kap'])).reshape(-1,4)[:nb].mean(1)
dH=np.array([(o1[k][0]-o0[k][0]).double().numpy().mean(0) for k in range(nb)]); dO=np.array([(o1[k][1]-o0[k][1]).double().numpy() for k in range(nb)])
pi=dH@TY['ins']; pn=dH@TY['ne']; offt=np.linalg.norm(dH-np.outer(pi,TY['ins'])-np.outer(pn,TY['ne']),axis=1)
for k in range(nb):
    if upres[k] or abs(pi[k])>.5 or abs(pn[k])>.5: print(k,'u',bool(upres[k]),'dec ins/ne',round(float(dec_b['ins'][k]),2),round(float(dec_b['ne'][k]),2),'| dy.ins',round(float(pi[k]),2),'dy.ne',round(float(pn[k]),2),'off',round(float(offt[k]),2),'|dout|',round(float(np.linalg.norm(dO[k])),3))
print('no-action max |dy.ins|,|dy.ne|,off:', round(float(np.abs(pi[~upres[:nb]]).max()),3), round(float(np.abs(pn[~upres[:nb]]).max()),3), round(float(offt[~upres[:nb]].max()),3))
```