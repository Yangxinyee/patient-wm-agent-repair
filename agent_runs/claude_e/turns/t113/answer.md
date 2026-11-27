The read-outs come out about as expected:
- **Insulin** (glucose rows): cross-validated R² 0.40, gain 0.40 (finger 0.43 / 0.41, lab 0.22 / 0.37).
- **NE** (MAP rows): R² 0.17, gain 0.18.
- **Gate:** a discriminant direction separates the code groups cleanly. Glucose codes score 1.68 to 1.84, MAP codes −0.92 to −0.76, so a steep gate (slope K about 50) cleanly switches each correction on for its own codes only.

**Concern:** the ridge chose the smallest penalty in its grid (0.01), giving the insulin read-out direction a large norm (83). A direction that relies on low-variance features may respond erratically to the large state changes of a doubled dose. I'll check how much R² a stronger penalty costs.

At this R², my formula puts the zero-cost insulin coefficient near −1.2. The NE side is expensive. Its read-out is noisy and MAP prediction errors are small (about 4.5 mmHg), so even a ≈ +2 might cost about 0.0025 nats overall. I'll compute the real costs with the local head instead of relying on formulas.

This run does two things:
1. The ridge penalty path for both read-outs: R² and norm against penalty.
2. The importance of each of the 192 value-head hidden units on the 500 train stays: the NLL change when its output column is zeroed. I need four units to repurpose, and will take the least important.

The plan is supported if a stronger penalty keeps R² within about 0.02 and some units show near-zero importance (|ΔNLL| < 0.0005).

```python
from sklearn.linear_model import Ridge
for nm, mask, ycol, codes_ in [('ins', mins, 'dev1', [6,10]), ('ne', mne, 'nedev1', [7,5])]:
    Rm = RA_[mask].reset_index(drop=True); Wm = WA_[mask]
    cl = np.array([codes_.index(int(c))*17 + int(l) for c, l in zip(Rm.code.values, Rm.lvb.values)])
    F_ = np.c_[Wm, np.eye(17*len(codes_))[cl]]; y = Rm[ycol].values
    res = []
    for al in [0.001, 0.01, 0.1, 1, 10, 100]:
        pred = cross_val_predict(Ridge(alpha=al), F_, y, cv=GroupKFold(5), groups=Rm.stay.values)
        wn = np.linalg.norm(Ridge(alpha=al).fit(F_, y).coef_[:192])
        res.append((al, round(1-np.mean((y-pred)**2)/y.var(),3), round(np.polyfit(y, pred, 1)[0],3), round(wn,1)))
    print(nm, 'alpha, R2, gain, |w|:', res)
# unit importance on 500 train stays
W0o = T('attr_heads.value_bin.0.weight'); b0o = T('attr_heads.value_bin.0.bias'); W2o = T('attr_heads.value_bin.2.weight'); b2o = T('attr_heads.value_bin.2.bias')
TRo = T('embed_transition.value_bin.weight')
imp_sum = np.zeros(192); base_sum = 0.0; ntot = 0
for fi, fn in enumerate(files):
    zz = np.load(fn, allow_pickle=True); Xs_ = torch.as_tensor(np.array(zz['x']), dtype=torch.float64)
    R_ = ALLR[fi]
    Q = q_from_rows(Xs_, torch.as_tensor(R_.prev.values.astype(int)), torch.as_tensor(R_.dt.values, dtype=torch.float64), nz=8, seed=5)
    f_ = reg_feats(torch.as_tensor(R_.has.values), torch.as_tensor(R_.age.values), torch.as_tensor(R_.zl.values))
    code = torch.as_tensor(R_.code.values.astype(int)); lvb = torch.as_tensor(R_.lvb.values.astype(int))
    lg, wc, hid = value_logits(Q, code, lvb, f_)
    ob = torch.as_tensor(R_.obin.values.astype(int)); ar = torch.arange(len(R_))
    p = torch.softmax(lg, -1).mean(0); base = -torch.log(p[ar, ob]); base_sum += base.sum().item(); ntot += len(R_)
    for j in range(192):
        lgj = lg - hid[..., j:j+1] * W2o[:, j][None, None, :]
        pj = torch.softmax(lgj, -1).mean(0)
        imp_sum[j] += (-torch.log(pj[ar, ob]) - base).sum().item()
imp_tr = imp_sum / ntot
order = np.argsort(np.abs(imp_tr))
print('base NLL (500 train stays):', base_sum/ntot)
print('least |importance| units:', order[:10].tolist(), np.round(imp_tr[order[:10]], 5).tolist())
print('run-84 candidates [68,123,46,118] on train:', np.round(imp_tr[[68,123,46,118]], 5).tolist())
```