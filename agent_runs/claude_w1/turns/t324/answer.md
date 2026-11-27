The new constraint shrank the decoders' between-cell spread as intended (total sd 0.81 for insulin, 0.21 for NE). It also opened a leak. The decoders are not orthogonal to the code embeddings, so on non-target codes the decoder term is large, and with the now-larger scale k that pushes those rows into the "on" regime (off_max 43 and 323). The non-target shifts (−45 to +103) and the wrong-signed direct effects follow from that.

Fix:
- Add the code embeddings (centred), and the rate-bin rows for completeness, to the constraint set. The decoder output then does not depend on which code is being predicted.
- Set k from the 99.99th percentile of the decoder term on target rows, so the term stays within about ±4.
- Rebuild, and refuse to save unless target rows stay above 3 and non-target rows below −4.

Success means lab and cuff shifts of about 0, target mean shifts of about 0 with moderate spread, and direct effects with the intended signs.

```python
cols2 = list(cols)
for n in ['embed.code.weight', 'embed.bins.rate.weight']:
    E_ = np.asarray(load_original(n), np.float64); cols2.append((E_ - E_.mean(0)).T)
U_, s_, _ = np.linalg.svd(np.concatenate(cols2, axis=1), full_matrices=False); Qr2 = U_[:, s_ > 1e-6 * s_[0]]
Pr2 = np.eye(192) - Qr2 @ Qr2.T
print('constraint dims', Qr2.shape[1])
DECR2 = {}; SLOPER2 = {}
for drug, con in [('insulin', 0), ('norepinephrine', 2)]:
    gi = np.concatenate([EDR[(drug, f)]['gi'] for f in (2.0, 0.5)])
    dX = np.concatenate([EDR[(drug, f)]['dX'] for f in (2.0, 0.5)]).astype(np.float64)
    dd = np.concatenate([EDR[(drug, f)]['dd'] for f in (2.0, 0.5)]).astype(np.float64)
    re = np.concatenate([EDR[(drug, f)]['reached'] for f in (2.0, 0.5)]).astype(bool)
    sel = re & (cf[gi] == con); dXs = dX[sel] @ Pr2; y = dd[sel, 0:3].mean(1)
    onidx = np.where(cf == con)[0]; R = cellres(onidx); S = Pr2 @ (R.T @ R / len(R)) @ Pr2
    for kap in [0.1, 0.3, 1.0]:
        v = Pr2 @ np.linalg.solve(dXs.T @ dXs / len(y) + kap * S + 1e-9 * np.eye(192), dXs.T @ y / len(y))
        r_ = dX[sel] @ v; sl = (r_ @ y) / (y @ y); noise = np.sqrt(v @ S @ v)
        sds = [(X64[cf == c] @ v).std() for c in range(4)]; mns = [(X64[cf == c] @ v).mean() for c in range(4)]
        DECR2[(drug, kap)] = v; SLOPER2[(drug, kap)] = sl
        print(drug, 'kap %g slope %.3f noise %.3f per-unit %.3f | code means ' % (kap, sl, noise, noise / sl) + ' '.join('%.2f' % m for m in mns) + ' sds ' + ' '.join('%.2f' % s for s in sds))
def build_v4(gI, gN, decI, decN, slI, slN, lamI=0.01, lamN=0.03, units=units4, save=True):
    W1 = W1o.copy(); b1 = b1o.copy(); W2 = W2o.copy(); TR = TRo.copy(); info = {}; ok = True
    for drug, con, g_, v, sl, lamf, (jA, jB), cname in [('insulin', 0, gI, decI, slI, lamI, units[0:2], 'glucose_finger'),
                                                        ('norepinephrine', 2, gN, decN, slN, lamN, units[2:4], 'map_arterial')]:
        c = g_ / sl; wg, lo_on, hi_off = GATE[(con, lamf)]
        a = 20.0 / (lo_on - hi_off); b = 10.0 - a * lo_on
        on = np.where(cf == con)[0]; vx = X64[on] @ v; mu = vx.mean()
        k = 4.0 / np.percentile(np.abs(vx - mu), 99.99)
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
        onmin = min(s[on].min(), preA[on].min()); offmax = max(s[offm].max(), preA[offm].max())
        ok = ok and onmin > 3 and offmax < -4
        info[drug] = dict(c=round(c, 3), k=round(k, 3), on_min=round(onmin, 2), off_max=round(offmax, 2))
    if save and ok:
        for nm, arr in [('attr_heads.value_bin.0.weight', W1), ('attr_heads.value_bin.0.bias', b1),
                        ('attr_heads.value_bin.2.weight', W2), ('embed_transition.value_bin.weight', TR)]:
            save_edit(nm, arr.astype(np.float32))
    info['saved'] = bool(save and ok)
    return W1, b1, W2, TR, info
kI, kN = 0.3, 0.3
W1e, b1e, W2e, TRe, info = build_v4(-2.0, 2.5, DECR2[('insulin', kI)], DECR2[('norepinephrine', kN)], SLOPER2[('insulin', kI)], SLOPER2[('norepinephrine', kN)])
print(info); check(W1e, b1e, W2e, TRe); direct(W1e, b1e, W2e, TRe)
print('edited:', edited())
```