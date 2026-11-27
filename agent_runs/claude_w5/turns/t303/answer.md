This check gives no clear driver. The 'factual' spread across edits only reflects glucose versus MAP readouts (my mistake), and the hour-13 gap correlates weakly with every bundle feature tried.

I now think the most likely explanation is that the dose_edit rollout runs numerically differently from states mode, for example batched or in reduced precision. Any such difference would break my stage-1 kv edit, which has weights near 2,000 and token values up to 2e4 that must cancel almost exactly. The original model, with ordinary magnitudes, is barely affected. That fits all the evidence:
- stay-specific deviations only for the edited model;
- noisy, biased dose responses;
- an even component, x1/2 with the wrong sign.

Whatever the exact cause, the robust fix is the same: remove the huge stage-1 edit. Stage 2 would then read the dose residual directly from n3 under the original attention, where slots attend about 100% to u at drug bundles. That works only if n3 without stage 1 still carries the rate.

So I'll test, under the true original weights, whether (rate − c_hat) is linearly decodable from slot-mean n3 at drug bundles. I'll use factual, x2 and x1/2 variants and held-out stays, and also check the drug versus non-drug gate.
- **Viable:** held-out R2 ≳ 0.8 and symmetric slopes (≈ +1 for x2, ≈ −0.5 for x1/2). I can then re-point the three FFN units to the new decoders and drop the kv edit.
- **Not viable:** R2 is low, and I need another route.

```python
import numpy as np, time
from sklearn.linear_model import Ridge
rng_ = np.random.default_rng(1); sids_s = list(rng_.choice(SIDS, 500, replace=False))
D_ = {'I': [], 'N': []}; NG = []; t0 = time.time()
for si, sid in enumerate(sids_s):
    Bs, Ss = bundles_B[sid], steps_all[sid]
    for j,(t,ats) in enumerate(Bs):
        k0 = [a for a in ats if a.get('kind')==0 and a['code'] in ('insulin','norepinephrine')]
        if not k0:
            if j % 4 == 0: NG.append(internals(W64, Ss[j])['n3'][:4].mean(0))
            continue
        st = np.concatenate([np.asarray(Ss[j]['out0'],float).ravel(), np.asarray(Ss[j]['h0'],float).reshape(-1,192).mean(0)])
        for drug in sorted(set(a['code'] for a in k0)):
            rate = [a['rate'] for a in k0 if a['code']==drug][0]; X_ = 'I' if drug=='insulin' else 'N'
            for f in [1.0, 2.0, 0.5]:
                if f == 1.0: d2 = Ss[j]
                else:
                    e2,u2 = enc2(scale_atoms(ats, drug, f)); d2 = dict(Ss[j]); d2['u'] = u2
                D_[X_].append((si, f, rate, internals(W64, d2)['n3'][:4].mean(0), st))
    if time.time()-t0 > 600: print('stop', si); break
print('time %.0f' % (time.time()-t0), {k: len(v) for k,v in D_.items()}, 'nondrug', len(NG))
for X_ in ['I','N']:
    si_ = np.array([r[0] for r in D_[X_]]); ff = np.array([r[1] for r in D_[X_]]); rr = np.array([r[2] for r in D_[X_]])
    N3x = np.stack([r[3] for r in D_[X_]]); ST = np.stack([r[4] for r in D_[X_]])
    tr = si_ % 5 != 0; m1 = ff==1.0
    ch = Ridge(alpha=100).fit(ST[tr & m1], rr[tr & m1]); chat = ch.predict(ST)
    tgt = rr*ff - chat
    for lam in [0.1, 1.0, 10.0]:
        dec = Ridge(alpha=lam).fit(N3x[tr], tgt[tr]); px = dec.predict(N3x)
        te = ~tr; r2 = 1 - ((px[te]-tgt[te])**2).sum()/((tgt[te]-tgt[te].mean())**2).sum()
        base_ = {}
        for f in [2.0, 0.5]:
            mf, mb = te & (ff==f), te & (ff==1.0)
            dxf = px[mf] - px[mb]; drf = rr[mf]*(f-1)
            base_[f] = (dxf*drf).sum()/(drf*drf).sum()
        print(X_, 'lam', lam, 'R2 test %.3f' % r2, 'slope x2 %.3f  x1/2 %.3f' % (base_[2.0], base_[0.5]), '|w| %.1f' % np.linalg.norm(dec.coef_), 'R2 c_hat %.3f' % ch.score(ST[~tr & m1], rr[~tr & m1]))
    globals()['DEC_'+X_] = dec; globals()['CH_'+X_+'_new'] = ch; globals()['N3_'+X_] = (si_, ff, rr, N3x, ST)
```