The cancelling-unit gate behaves exactly as designed. Non-target codes (lab, cuff) shift by at most 0.06, and target rows stay in the linear regime (minimum pre-activation about 9; non-target rows at most −7). My logit formula matches the kernel's exactly, so the code mapping is confirmed.

But the target-code shifts are far too noisy: standard deviation about 10 and maximum 44–62, with a bias of −1.7. The cause is the decoder output's large variation between last-bin cells (overall sd about 97, against about 0.8 within cells). Shrinking small cells' means toward the global mean therefore mis-centres them badly. A second cause is that a logit tilt shifts the mean in proportion to each row's predictive variance.

Fixes:
1. Constrain the decoders to also be orthogonal to the register embeddings (last_numeric columns, bins.value rows, last_status rows), so they carry no last-value, age or bin dependence.
2. Centre each cell exactly, with the mean weighted by predictive variance. That is the KL-optimal centring and makes each cell's mean shift zero.

This run should show replica shift means near 0 and much smaller spreads.

```python
bn = [n for n in NAMES if ('bins' in n or 'last_status' in n or n.startswith('embed.code')) and n.endswith('weight')]
print([(n, SHAPES[n]) for n in bn])
cols = [Qz, np.asarray(load_original('last_numeric.weight'), np.float64)]
for n in bn:
    if 'bins.value' in n or 'last_status' in n:
        E_ = np.asarray(load_original(n), np.float64); cols.append((E_ - E_.mean(0)).T)
Qr, sv = np.linalg.qr(np.concatenate(cols, axis=1))[0], None
U_, s_, _ = np.linalg.svd(np.concatenate(cols, axis=1), full_matrices=False); Qr = U_[:, s_ > 1e-6 * s_[0]]
Pr = np.eye(192) - Qr @ Qr.T
print('constraint dims', Qr.shape[1])
DECR = {}; SLOPER = {}
for drug, con in [('insulin', 0), ('norepinephrine', 2)]:
    gi = np.concatenate([EDR[(drug, f)]['gi'] for f in (2.0, 0.5)])
    dX = np.concatenate([EDR[(drug, f)]['dX'] for f in (2.0, 0.5)]).astype(np.float64)
    dd = np.concatenate([EDR[(drug, f)]['dd'] for f in (2.0, 0.5)]).astype(np.float64)
    re = np.concatenate([EDR[(drug, f)]['reached'] for f in (2.0, 0.5)]).astype(bool)
    sel = re & (cf[gi] == con); dXs = dX[sel] @ Pr; y = dd[sel, 0:3].mean(1)
    onidx = np.where(cf == con)[0]; R = cellres(onidx); S = Pr @ (R.T @ R / len(R)) @ Pr
    for kap in [0.1, 0.3, 1.0]:
        v = Pr @ np.linalg.solve(dXs.T @ dXs / len(y) + kap * S + 1e-9 * np.eye(192), dXs.T @ y / len(y))
        r_ = dX[sel] @ v; sl = (r_ @ y) / (y @ y); noise = np.sqrt(v @ S @ v); vx = X64[onidx] @ v
        DECR[(drug, kap)] = v; SLOPER[(drug, kap)] = sl
        print(drug, 'kap %g slope %.3f noise %.3f per-unit %.3f  total sd %.3f' % (kap, sl, noise, noise / sl, vx.std()))
def build_v3(gI, gN, decI, decN, slI, slN, lamI=0.01, lamN=0.03, units=units4, save=True):
    W1 = W1o.copy(); b1 = b1o.copy(); W2 = W2o.copy(); TR = TRo.copy(); info = {}
    for drug, con, g_, v, sl, lamf, (jA, jB), cname in [('insulin', 0, gI, decI, slI, lamI, units[0:2], 'glucose_finger'),
                                                        ('norepinephrine', 2, gN, decN, slN, lamN, units[2:4], 'map_arterial')]:
        c = g_ / sl; wg, lo_on, hi_off = GATE[(con, lamf)]
        a = 20.0 / (lo_on - hi_off); b = 10.0 - a * lo_on
        on = np.where(cf == con)[0]; vx = X64[on] @ v; mu = vx.mean(); k = 1.0 / max(vx.std(), 1e-9)
        W1[jA] = a * wg + k * v; b1[jA] = b - k * mu; W1[jB] = a * wg; b1[jB] = b
        C = cenof(cname); tv = np.zeros(17); tv[1:] = C - C.mean(); ci = cidx(cname)
        L0 = gel_(X64[on] @ W1o.T + b1o) @ W2o.T + b2o + TRo[ci * 17 + lbf[on]]
        P = smx(L0); P1 = P[:, 1:] / P[:, 1:].sum(1, keepdims=True); Vp = (P1 * C ** 2).sum(1) - (P1 @ C) ** 2
        medvar = np.median(Vp)
        W2[:, jA] = (c / k) / medvar * tv; W2[:, jB] = -(c / k) / medvar * tv
        mw = (Vp * vx).sum() / Vp.sum()
        for bb in range(17):
            m = lbf[on] == bb
            mb = (Vp[m] * vx[m]).sum() / Vp[m].sum() if m.sum() > 0 else mw
            TR[ci * 17 + bb] -= (c / medvar) * tv * (mb - mu)
        s = a * (X64 @ wg) + b; preA = s + k * (X64 @ v - mu); offm = cf != con
        info[drug] = dict(c=round(c, 3), on_min=round(min(s[on].min(), preA[on].min()), 2), off_max=round(max(s[offm].max(), preA[offm].max()), 2))
    if save:
        for nm, arr in [('attr_heads.value_bin.0.weight', W1), ('attr_heads.value_bin.0.bias', b1),
                        ('attr_heads.value_bin.2.weight', W2), ('embed_transition.value_bin.weight', TR)]:
            save_edit(nm, arr.astype(np.float32))
    return W1, b1, W2, TR, info
def check(W1e, b1e, W2e, TRe):
    out = []
    for c in range(4):
        idx = np.where(cf == c)[0]; C = cenof(names4[c])
        L0 = gel_(X64[idx] @ W1o.T + b1o) @ W2o.T + b2o + TRo[cid4[c] * 17 + lbf[idx]]
        L1 = gel_(X64[idx] @ W1e.T + b1e) @ W2e.T + b2o + TRe[cid4[c] * 17 + lbf[idx]]
        P0 = smx(L0); P1 = smx(L1)
        d = (P1[:, 1:] / P1[:, 1:].sum(1, keepdims=True)) @ C - (P0[:, 1:] / P0[:, 1:].sum(1, keepdims=True)) @ C
        kl = (P0 * (np.log(P0 + 1e-12) - np.log(P1 + 1e-12))).sum(1)
        out.append('%s mean %.3f sd %.3f max %.2f KL %.4f' % (names4[c][-6:], d.mean(), d.std(), np.abs(d).max(), kl.mean()))
    print(' | '.join(out))
global lbf_tmp
def direct(W1e, b1e, W2e, TRe):
    global lbf_tmp
    res = []
    for drug, con in [('insulin', 0), ('norepinephrine', 2)]:
        for f in (2.0, 0.5):
            E = EDR[(drug, f)]; sel = np.where(E['reached'].astype(bool) & (cf[E['gi']] == con))[0][:4000]
            gi = E['gi'][sel]; X0 = X64[gi]; X1 = X0 + E['dX'][sel].astype(np.float64); lbf_tmp = lbf[gi]
            de = (emean(X1, con, W1e, b1e, W2e, TRe) - emean(X1, con, W1o, b1o, W2o, TRo)) - (emean(X0, con, W1e, b1e, W2e, TRe) - emean(X0, con, W1o, b1o, W2o, TRo))
            res.append('%s%.1f %.3f' % (drug[:3], f, de.mean()))
    print('direct:', ' | '.join(res))
kI, kN = 0.3, 0.3
W1e, b1e, W2e, TRe, info = build_v3(-2.0, 2.5, DECR[('insulin', kI)], DECR[('norepinephrine', kN)], SLOPER[('insulin', kI)], SLOPER[('norepinephrine', kN)])
print(info); check(W1e, b1e, W2e, TRe); direct(W1e, b1e, W2e, TRe)
print('edited:', edited())
```