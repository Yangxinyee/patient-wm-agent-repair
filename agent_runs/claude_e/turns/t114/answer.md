Two results shape the edit.

**1. Read-out precision vs. robustness.** Raising the ridge penalty costs R² steadily: insulin falls from 0.40 to 0.36 to 0.32 as the read-out norm drops from 83 to 31 to 10. NE falls from 0.17 to 0.16 to 0.14.

I'll use penalty 0.1 for both. That gives insulin R² 0.36 (norm 31) and NE 0.16 (norm 6). It keeps most of the accuracy while relying less on fine state directions, which may behave unpredictably under doubled doses.

**2. Spare units.** Units 26, 41, 131 and 159 of the value head have essentially zero effect on the 500 train stays (|ΔNLL| ≤ 1e-5). I'll repurpose those. Some other units seem to hurt the local NLL slightly, but I won't remove them to bank budget. That would be optimising the head against the NLL, and might reflect small gaps in my reconstruction.

**The edit, as specified so far:**
- Two code-gated GELU pairs, using the LDA gate with slope K = 40.
- Insulin read-out scaled to unit gain, times a_ins, as a logit tilt over the glucose bins.
- NE read-out the same way, times a_ne, over the MAP bins.
- Per-(code, last-bin) read-out offsets added through the transition table.

For a first test I take a_ins = −1.5 mg/dL per unit/h, just above the cost-free level, and a_ne = +2.0 mmHg per mcg/kg/min.

Before saving anything, I check offline with the local head on the 500 train stays and the 100 sel1 tuning stays:
- the NLL change by event;
- that the gate leaves the other code group exactly untouched;
- that the realised mean shift tracks the intended shift.

If the overall NLL change is ≤ +0.002 and the mean-shift correlation is high, I save the edit and test it with the forward tool. If the cost is too high, I shrink the coefficients first.

```python
def refit(mask, ycol, codes_, alpha):
    Rm = RA_[mask].reset_index(drop=True); Wm = WA_[mask]
    cl = np.array([codes_.index(int(c))*17 + int(l) for c, l in zip(Rm.code.values, Rm.lvb.values)])
    F_ = np.c_[Wm, np.eye(17*len(codes_))[cl]]; y = Rm[ycol].values
    pred = cross_val_predict(Ridge(alpha=alpha), F_, y, cv=GroupKFold(5), groups=Rm.stay.values)
    m = Ridge(alpha=alpha).fit(F_, y)
    return dict(w=m.coef_[:192], b=m.coef_[192:].reshape(len(codes_),17), w0=m.intercept_, gain=np.polyfit(y, pred, 1)[0], r2=1-np.mean((y-pred)**2)/y.var(), codes=codes_)
RO_i = refit(mins, 'dev1', [6,10], 0.1); RO_n = refit(mne, 'nedev1', [7,5], 0.1)
print('ins', round(RO_i['r2'],3), round(RO_i['gain'],3), ' ne', round(RO_n['r2'],3), round(RO_n['gain'],3))
print('centres equal 6/10:', torch.allclose(CENT[6], CENT[10]), ' 7/5:', torch.allclose(CENT[7], CENT[5]))
gelu_ = torch.nn.functional.gelu
def head_run(Xs_, R_, Wd, nz=16, seed=0):
    Q = q_from_rows(Xs_, torch.as_tensor(R_.prev.values.astype(int)), torch.as_tensor(R_.dt.values, dtype=torch.float64), nz=nz, seed=seed)
    f_ = reg_feats(torch.as_tensor(R_.has.values), torch.as_tensor(R_.age.values), torch.as_tensor(R_.zl.values))
    code = torch.as_tensor(R_.code.values.astype(int)); lvb = torch.as_tensor(R_.lvb.values.astype(int))
    wc = Q + (T('embed.code.weight')[code] + T('embed.bins.value.weight')[lvb] + f_ @ T('last_numeric.weight').T + T('last_numeric.bias'))[None]
    hid = gelu_(wc @ Wd['W0'].T + Wd['b0'])
    lg = hid @ Wd['W2'].T + b2o + Wd['TR'][code*17 + lvb][None]
    p = torch.softmax(lg, -1).mean(0)
    cen = torch.stack([CENT[int(c)] for c in R_.code.values])
    pm = (p[:,1:]*cen).sum(1); nll = -torch.log(p[torch.arange(len(R_)), torch.as_tensor(R_.obin.values.astype(int))])
    return pm.numpy(), nll.numpy(), p.numpy(), hid
ORIG = dict(W0=W0o, b0=b0o, W2=W2o, TR=TRo)
# predictive variance of centres -> tilt vectors
datasets = [(torch.as_tensor(np.array(np.load(fn, allow_pickle=True)['x']), dtype=torch.float64), ALLR[fi]) for fi, fn in enumerate(files)] + [(X if torch.is_tensor(X) else torch.as_tensor(X), RR)]
base_out = [head_run(Xs_, R_, ORIG, seed=3) for Xs_, R_ in datasets]
def vref(codes_, cond_col):
    vs = []
    for (Xs_, R_), (pm, nll, p, _) in zip(datasets[:5], base_out[:5]):
        m_ = R_.code.isin(codes_).values & (np.abs(R_[cond_col].values) > 1e-6 if cond_col in R_ else True)
        cen = torch.stack([CENT[int(c)] for c in R_.code.values[m_]]).numpy()
        pp = p[m_, 1:] / p[m_, 1:].sum(1, keepdims=True)
        mu = (pp*cen).sum(1); vs.append((pp*cen**2).sum(1) - mu**2)
    v = np.concatenate(vs); return np.median(v), np.percentile(v, [10, 90])
for R_ in ALLR: R_['dev1'] = RA_.loc[RA_.file == R_.file.iloc[0], 'dev1'].values; R_['nedev1'] = RA_.loc[RA_.file == R_.file.iloc[0], 'nedev1'].values
Vg, Vg_q = vref([6], 'dev1'); Vm, Vm_q = vref([7], 'nedev1')
print('V_glu median', Vg, Vg_q, ' V_map median', Vm, Vm_q)
cg_ = CENT[6].numpy(); cm_ = CENT[7].numpy()
tg = torch.zeros(17, dtype=torch.float64); tg[1:] = torch.as_tensor((cg_ - cg_.mean())/Vg)
tm = torch.zeros(17, dtype=torch.float64); tm[1:] = torch.as_tensor((cm_ - cm_.mean())/Vm)
units = [26, 41, 131, 159]; mid = 0.46; gvec = torch.as_tensor(glda)
def make_edit(a_ins, a_ne, K=40.0):
    W0 = W0o.clone(); b0 = b0o.clone(); W2 = W2o.clone(); TR = TRo.clone()
    wi = torch.as_tensor(RO_i['w'])/RO_i['gain']; w0i = RO_i['w0']/RO_i['gain']
    wn = torch.as_tensor(RO_n['w'])/RO_n['gain']; w0n = RO_n['w0']/RO_n['gain']
    u1, u2, u3, u4 = units
    W0[u1] = K*gvec + wi; b0[u1] = -K*mid + w0i; W0[u2] = K*gvec; b0[u2] = -K*mid
    W0[u3] = -K*gvec + wn; b0[u3] = K*mid + w0n; W0[u4] = -K*gvec; b0[u4] = K*mid
    W2[:, u1] = a_ins*tg; W2[:, u2] = -a_ins*tg; W2[:, u3] = a_ne*tm; W2[:, u4] = -a_ne*tm
    for ci, c in enumerate(RO_i['codes']):
        for l in range(17): TR[c*17+l] = TR[c*17+l] + a_ins*(RO_i['b'][ci, l]/RO_i['gain'])*tg
    for ci, c in enumerate(RO_n['codes']):
        for l in range(17): TR[c*17+l] = TR[c*17+l] + a_ne*(RO_n['b'][ci, l]/RO_n['gain'])*tm
    return dict(W0=W0, b0=b0, W2=W2, TR=TR)
ED = make_edit(-1.5, 2.0)
tot = {}
for di, ((Xs_, R_), (pm0, nll0, p0, hid0)) in enumerate(zip(datasets, base_out)):
    pm1, nll1, p1, hid1 = head_run(Xs_, R_, ED, seed=3)
    for ev in R_.event.unique():
        m_ = (R_.event == ev).values
        tot.setdefault(('train' if di < 5 else 'sel1', ev), []).append((nll1[m_]-nll0[m_]).sum()); tot.setdefault(('train' if di < 5 else 'sel1', ev+'_n'), []).append(m_.sum())
        tot.setdefault(('train' if di < 5 else 'sel1', ev+'_shift'), []).append((pm1[m_]-pm0[m_]).sum())
    if di == 0:
        g_ = R_.code.isin([6,10]).values; mp = ~g_
        print('gate check: max |unit pair diff| on MAP rows (ins pair):', float((hid1[:, mp, units[0]]-hid1[:, mp, units[1]]).abs().max()), ' on glucose rows (ne pair):', float((hid1[:, g_, units[2]]-hid1[:, g_, units[3]]).abs().max()))
        print('min unit2 preact glucose rows:', float((hid1[:, g_, units[1]]).min()), ' corr(shift, a*dev1) gf:', np.corrcoef(pm1[R_.code.values==6]-pm0[R_.code.values==6], R_.dev1.values[R_.code.values==6])[0,1])
for split in ['train', 'sel1']:
    tn = sum(sum(v) for k, v in tot.items() if k[0]==split and not k[1].endswith('_n') and not k[1].endswith('_shift'))
    nn = sum(sum(v) for k, v in tot.items() if k[0]==split and k[1].endswith('_n'))
    print(split, 'overall dNLL per reading', round(tn/nn, 5), {k[1]: round(sum(v)/sum(tot[(split, k[1]+'_n')]), 5) for k, v in tot.items() if k[0]==split and not k[1].endswith('_n') and not k[1].endswith('_shift')})
    print('   mean pm shift by event', {k[1][:-6]: round(sum(v)/sum(tot[(split, k[1][:-6]+'_n')]), 3) for k, v in tot.items() if k[0]==split and k[1].endswith('_shift')})
```