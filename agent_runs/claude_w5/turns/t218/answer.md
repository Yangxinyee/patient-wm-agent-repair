Both decoders read the rate well from the prelude FFN input n3 (validation R² 0.98 insulin, 0.91 norepinephrine), with little cross-talk between the drugs. Under actual dose scaling they respond with slope about 0.9 for both x2 and x½, so one gain per drug can fit both readouts. The per-bundle noise in that response is moderate (R² 0.5–0.8), and it averages out in the pooled readout. The u-token decoder breaks for norepinephrine x2 (slope -0.05), so n3 is the right site. The policy mean ĉ predicted from the state has R² 0.77 (insulin) and 0.49 (norepinephrine).

Plan: a few repurposed prelude-FFN units compute x = rate − ĉ directly from n3, open only at drug bundles. Next I fit:
1. the residual decoder for x, on factual plus x2/x½ samples so it stays accurate across the edited range;
2. a drug-versus-non-drug discriminant from n3 over all 5,294 bundles.

The gate works only if the discriminant has a clear margin on held-out stays, including at scaled-dose bundles, and the decoded x stays bounded at non-drug bundles.

```python
# n3 for all bundles (gate data)
N3all, ISD, SIDn = [], [], []
for s in sids_B:
    for k, d in enumerate(steps_all[s]):
        R = {}; mystep(d, R=R); N3all.append(R['n3'][0].numpy()); ISD.append(bool((d['u'] - u_empty).norm() > 1e-6)); SIDn.append(s)
N3all = np.array(N3all); ISD = np.array(ISD); trn = np.array([s in sid_tr for s in SIDn])
# c_hat from state (lam 100), fitted on train-stay drug bundles
cI_w, cI_b = ridge_fit(ST, YI, 100.0); cN_w, cN_b = ridge_fit(ST, YN, 100.0)
CH_I = ST@cI_w + cI_b; CH_N = ST@cN_w + cN_b
# residual decoder dataset: factual + scaled
Xr, Yr, Tr_, Kd = [], {'I': [], 'N': []}, [], []
for i, r in enumerate(DB):
    for key, x in r['n3'].items():
        rI = r['rI'] * (key[1] if key != 1.0 and key[0]=='I' else 1.0); rN = r['rN'] * (key[1] if key != 1.0 and key[0]=='N' else 1.0)
        Xr.append(x); Yr['I'].append(rI - CH_I[i]); Yr['N'].append(rN - CH_N[i]); Tr_.append(tr[i]); Kd.append(0 if key == 1.0 else 1)
Xr = np.array(Xr); Tr_ = np.array(Tr_); Kd = np.array(Kd)
def ridge2(X, y, m, lam):
    mu = X[m].mean(0); Xc = X[m]-mu; ym = y[m].mean()
    w = np.linalg.solve(Xc.T@Xc + lam*np.eye(X.shape[1]), Xc.T@(y[m]-ym)); return w, ym - mu@w
XD = {}
for tg in ('I','N'):
    y = np.array(Yr[tg])
    for lam in (1e-4, 1e-3, 1e-2):
        w, b = ridge2(Xr, y, Tr_, lam); yh = Xr@w + b; v = ~Tr_
        print(tg, 'lam', lam, 'val R2 factual', round(r2(yh[v & (Kd==0)], y[v & (Kd==0)]),3), 'scaled', round(r2(yh[v & (Kd==1)], y[v & (Kd==1)]),3),
              'resid sd fact', round((yh-y)[v&(Kd==0)].std(),3), 'non-drug |x| max', round(np.abs(N3all[~ISD]@w+b).max(),2))
        XD[(tg, lam)] = (w, b)
# gate: drug (incl scaled) vs non-drug
Xg = np.vstack([N3all[~ISD], Xr]); yg = np.r_[-np.ones((~ISD).sum()), np.ones(len(Xr))]
mg = np.r_[trn[~ISD], Tr_]
for lam in (1e-3, 1e-2, 1e-1):
    w, b = ridge2(Xg, yg, mg, lam); sg_ = Xg@w + b; v = ~mg
    print('gate lam', lam, 'val drug min', round(sg_[v & (yg>0)].min(),3), 'non max', round(sg_[v & (yg<0)].max(),3),
          '| train drug min', round(sg_[mg & (yg>0)].min(),3), 'non max', round(sg_[mg & (yg<0)].max(),3))
    XD[('gate', lam)] = (w, b)
```