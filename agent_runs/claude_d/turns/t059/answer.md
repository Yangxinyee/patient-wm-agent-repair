The Hansen J test settles the NE shape. Linear in rate (p = 1) fits best, with J = 21.3 on 16 df, acceptable. The log coding is strongly rejected (J = 582), and p = 0.75 and p = 1.5 fit worse.

The clean estimate is γ = [3.28, 2.73, 2.02, 1.21, 0.43, 0.07, −0.16, 0.23] mmHg per mcg/kg/min at lags 1–8. That almost reproduces my original gam_ne ([3.16, 2.63, 1.78, 0.95, 0.46, …]). So the "log-linear" scare came from the flawed EPSN subset, and both drugs act linearly and additively.

Next I'll compute the four true targets on the 100 tuning stays of forward pass 1, using exactly the reached stay-hours from the dose_edit CSV. The shift at hour t is the sum over k of coef_k × (factor − 1) × dose_{t−k}, counting only doses from hour 12 on. I'll compare these with the model's current readouts and save the estimates.

```python
gam_clean = np.concatenate([[0.0], b]) if False else None
Ap = np.where(NOR>0, NOR, 0.0); b_ne, J_ = iv2(MA, Ap, Ztr, strat, 8)
gam_clean = np.concatenate([[0.0], b_ne])
bet = np.array(json.load(open('/work/beta_ins.json'))) if isinstance(json.load(open('/work/beta_ins.json')), list) else np.array(json.load(open('/work/effects.json'))['beta_ins'])
bet = np.array(json.load(open('/work/effects.json'))['beta_ins'])
def true_shift(D, coef, fac, i, t):
    s=0.0
    for k in range(1, len(coef)):
        s_ = t-k
        if s_>=12: s += coef[k]*(fac-1)*D[i,s_]
    return s
tg={}
for e_, (D, coef, fac) in {'insulin_x2':(INS_U,bet,2.0),'insulin_half':(INS_U,bet,0.5),'ne_x2':(NOR_U,gam_clean,2.0),'ne_half':(NOR_U,gam_clean,0.5)}.items():
    rr_ = de[(de.edit==e_)&(de.reached)]
    ts = np.array([true_shift(D, coef, fac, sid2i[s], h) for s,h in zip(rr_.stay_id, rr_.hour)])
    tg[e_]=ts.mean()
    print('%-13s n=%d  TRUE(linear) %.2f   model %.3f'%(e_, len(rr_), ts.mean(), rr_['shift'].mean()))
ej = json.load(open('/work/effects.json')); ej['gam_ne_clean'] = gam_clean.tolist(); json.dump(ej, open('/work/effects.json','w'))
with open('/work/notes.md','a') as f: f.write("""
- Run 55: clean nonparametric NE instrument (rate - E[rate|exp-rate bin x MAP band], titration rows) + Hansen J over power codings r^p:
  p=1 (LINEAR) best J=21 (df16); log rejected (J=582). gam_ne_clean (lags1-8) = %s ~ same as old gam_ne. NE linear additive.
- Targets on sel1 reached hours (linear SNMM): %s
""" % (np.round(b_ne,2).tolist(), {k:round(v,2) for k,v in tg.items()}))
```