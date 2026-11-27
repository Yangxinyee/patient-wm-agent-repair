Fitting the rate readouts only where their drug is present works. The mean bias is within ±0.02 for every dose multiplier, and the spread is rms 0.03–0.14 for insulin and 0.02–0.09 for NE. That is small compared with the insulin surprise sd of 0.63, and it will not distort the ×0.5, ×2 or ×3 answers systematically. The presence gates handle the drug-absent bundles.

Now I'll build stage 1 of edit v2:
1. Write four carriers into x1 through head 1's value rows: insulin rate, insulin presence, NE rate and NE presence, with modest magnitudes (3, 2, 6 and 2 per unit).
2. Recompute the prelude exactly for the factual and dose-edited bundles.
3. Fit 5-parameter readouts of each carrier from n3.
4. Fit the expected insulin rate (band mean) from n3 with heavily weighted constraint rows, so it does not move when the current bundle's doses are edited.

Success means precise carrier readouts, an expected-rate CV R2 near 0.85, and a near-zero response of the expected rate to dose edits.

```python
CAR = ['tI','pI','tN','pN']; bsel = {'tI':BB['tI_p'], 'pI':BB['pI'], 'tN':BB['tN_p'], 'pN':BB['pN']}
adirs = [avg[:,i]/np.linalg.norm(Wo1@avg[:,i]) for i in range(4)]   # head-space vectors, Wo1@a unit norm
def make_kv(scales):
    kv_ = load_original('core.prelude.0.attn.kv.weight').clone()
    add = sum(scales[i]*np.outer(adirs[i], bsel[CAR[i]]) for i in range(4))
    kv_[240:288,:] += torch.tensor(add, dtype=torch.float32)
    return kv_
SC = [3.0, 2.0, 6.0, 2.0]
W_v2 = dict(W_); W_v2['core.prelude.0.attn.kv.weight'] = make_kv(SC)
cdirs = np.stack([Wo1@a for a in adirs])            # 4x192 unit carrier directions in x1
Vc = (w3[None,:]*(cdirs - cdirs.mean(1,keepdims=True))).T   # 192x4 in n3 space
RR = {k: prelude2(TOK[VAR[k]['selr']] if k!=(1,1) else TOK, W_v2) for k in []}
def tok_var(k):
    if k==(1,1): return TOK
    selr, T_ = tok_with(*k); return T_
N3 = {}; X1c = {}
for k in VAR:
    rr = prelude2(tok_var(k), W_v2); N3[k] = rr['n3'][:, :4, :].reshape(-1,192).numpy()
rrn = prelude2(TOKn, W_v2); N3n = rrn['n3'][:, :4, :].reshape(-1,192).numpy()
# carrier readouts from n3 (5 params each)
Z_all = np.concatenate([N3[k] for k in VAR] + [N3n]) @ Vc
RO = {}
for nm in CAR:
    y_ = TT[nm]; Xz = np.c_[Z_all, np.ones(len(Z_all))]
    c_ = np.linalg.lstsq(Xz, y_, rcond=None)[0]; RO[nm] = c_
    e_ = Xz@c_ - y_; pres = TT['pI' if nm in ('tI','pI') else 'pN'] > 0.5
    print(nm, 'readout rms(all) %.4f  rms(drug-present) %.4f  max(drug-absent) %.3f' % (np.sqrt((e_**2).mean()), np.sqrt((e_[pres]**2).mean()), np.abs(e_[~pres]).max()))
```