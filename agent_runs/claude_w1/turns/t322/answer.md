The z-orthogonal decoders work.
- Insulin: noise is about 1.3–1.5 per unit of response (slope 0.58–0.66, R² about 0.82).
- NE: noise is about 0.27–0.39 per unit (slope 0.72–0.76).
- The per-code gates separate the target code from the other three by 36 (finger) and 45 (arterial) within-code standard deviations.
- Eight hidden units are effectively dead (importance about 0), so they can be repurposed without touching the existing fit.

Next I build EDIT v2 and save it:
- Insulin correction on glucose_finger rows only, NE correction on map_arterial rows only.
- Each uses a cancelling unit pair: gate mapped so the target code sits at about +10 and every other code at about −10, plus a small decoder term.
- The logits are tilted along the bin centres, and each last-bin cell is centred through embed_transition.
- Initial effective gains are −2.0 mg/dL per u/h (insulin) and +2.5 mmHg per mcg/kg/min (NE).

Before saving I check three things:
1. My logit formula reproduces the kernel's head_logits, which confirms the code-index mapping into embed_transition.
2. Target rows stay in the linear regime and all other rows stay off.
3. The replica's predicted shifts: about zero mean per code, essentially zero on non-target codes, plus the size of the direct head effect of each dose edit.

Clean outputs here would confirm the construction, and the tool will then give the real cost and readouts.

```python
from scipy.special import erf
def smx(L):
    L = L - L.max(1, keepdims=True); e = np.exp(L); return e / e.sum(1, keepdims=True)
cds = VOCAB['codes']
print('codes:', str(cds)[:260])
def cidx(name):
    if isinstance(cds, dict):
        if name in cds:
            v = cds[name]
            if isinstance(v, (int, np.integer)): return int(v)
            if isinstance(v, dict):
                for kk in ('index', 'id', 'idx', 'code'):
                    if kk in v and isinstance(v[kk], (int, np.integer)): return int(v[kk])
            return list(cds.keys()).index(name)
        return None
    for i, c in enumerate(cds):
        if c == name: return i
        if isinstance(c, dict) and name in [str(x) for x in c.values()]:
            for kk in ('index', 'id', 'idx', 'code'):
                if kk in c and isinstance(c[kk], (int, np.integer)): return int(c[kk])
            return i
    return None
names4 = ['glucose_finger', 'glucose_lab', 'map_arterial', 'map_cuff']
cid4 = np.array([cidx(n) for n in names4]); print('cid4', cid4)
cen = VOCAB['value_bin_centres']
def cenof(name):
    a = np.asarray(cen[name] if isinstance(cen, dict) else cen[cidx(name)], np.float64)
    return a[1:] if len(a) == 17 else a
print('centre lens', [len(cenof(n)) for n in names4], 'finger', cenof('glucose_finger')[[0, -1]], 'art', cenof('map_arterial')[[0, -1]])
b2o = np.asarray(load_original('attr_heads.value_bin.2.bias'), np.float64)
TRo = np.asarray(load_original('embed_transition.value_bin.weight'), np.float64)
Xs = X64[rs]; cs_ = cf[rs]; lbs = lbf[rs]
Lm = Hs @ W2o.T + b2o + TRo[cid4[cs_] * 17 + lbs]
try:
    Lk = head_logits(Xf[rs], cs_, lbs)
    Lk = np.asarray(Lk[0] if isinstance(Lk, tuple) else Lk, np.float64)
    print('head_logits', Lk.shape, 'max|mine-kernel| %.2e' % np.abs(Lm - Lk).max())
except Exception as ex:
    print('head_logits err', str(ex)[:120])
SLOPE = {}
for drug, con in [('insulin', 0), ('norepinephrine', 2)]:
    gi = np.concatenate([EDR[(drug, f)]['gi'] for f in (2.0, 0.5)])
    dX = np.concatenate([EDR[(drug, f)]['dX'] for f in (2.0, 0.5)]).astype(np.float64)
    dd = np.concatenate([EDR[(drug, f)]['dd'] for f in (2.0, 0.5)]).astype(np.float64)
    re = np.concatenate([EDR[(drug, f)]['reached'] for f in (2.0, 0.5)]).astype(bool)
    sel = re & (cf[gi] == con); y = dd[sel, 0:3].mean(1)
    for kap in [0.1, 0.3]:
        r_ = dX[sel] @ DECZ[(drug, kap)]; SLOPE[(drug, kap)] = (r_ @ y) / (y @ y)
cand = order[:8]
crit = np.linalg.norm(W2o[:, cand], axis=0) * np.abs(Hs[:, cand]).max(0)
units4 = [int(u) for u in cand[np.argsort(crit)][:4]]
print('units4', units4, 'crit', np.sort(crit)[:4])
def build_v2(gI, gN, kapI=0.3, kapN=0.3, lamI=0.01, lamN=0.03, units=units4, save=True):
    W1 = W1o.copy(); b1 = b1o.copy(); W2 = W2o.copy(); TR = TRo.copy(); info = {}
    for drug, con, g_, kap, lamf, (jA, jB), cname in [('insulin', 0, gI, kapI, lamI, units[0:2], 'glucose_finger'),
                                                     ('norepinephrine', 2, gN, kapN, lamN, units[2:4], 'map_arterial')]:
        v = DECZ[(drug, kap)]; c = g_ / SLOPE[(drug, kap)]
        wg, lo_on, hi_off = GATE[(con, lamf)]
        a = 20.0 / (lo_on - hi_off); b = 10.0 - a * lo_on
        on = cf == con; vx = X64 @ v; mu = vx[on].mean(); sdv = vx[on].std(); k = 1.0 / sdv
        s = a * (X64 @ wg) + b; preA = s + k * (vx - mu)
        W1[jA] = a * wg + k * v; b1[jA] = b - k * mu; W1[jB] = a * wg; b1[jB] = b
        C = cenof(cname); tv = np.zeros(17); tv[1:] = C - C.mean()
        rows = np.where(cs_ == con)[0]; P = smx(Lm[rows]); P1 = P[:, 1:] / P[:, 1:].sum(1, keepdims=True)
        medvar = np.median((P1 * C ** 2).sum(1) - (P1 @ C) ** 2)
        W2[:, jA] = (c / k) / medvar * tv; W2[:, jB] = -(c / k) / medvar * tv
        ci = cidx(cname)
        for bb in range(17):
            m = on & (lbf == bb); n = m.sum(); mb = (vx[m].sum() + 20 * mu) / (n + 20)
            TR[ci * 17 + bb] -= (c / medvar) * tv * (mb - mu)
        info[drug] = dict(c=round(c, 3), k=round(k, 4), on_min=round(min(s[on].min(), preA[on].min()), 2),
                          off_max=round(max(s[~on].max(), preA[~on].max()), 2), medvar=round(medvar, 1), absW1=round(np.abs(W1[jA]).max(), 2))
    if save:
        for nm, arr in [('attr_heads.value_bin.0.weight', W1), ('attr_heads.value_bin.0.bias', b1),
                        ('attr_heads.value_bin.2.weight', W2), ('embed_transition.value_bin.weight', TR)]:
            try:
                save_edit(nm, arr.astype(np.float32))
            except Exception:
                save_edit(nm, torch.tensor(arr, dtype=torch.float32))
    return W1, b1, W2, TR, info
W1e, b1e, W2e, TRe, info = build_v2(-2.0, 2.5)
print(info)
He = gel_(Xs @ W1e.T + b1e); Le = He @ W2e.T + b2o + TRe[cid4[cs_] * 17 + lbs]
for c in range(4):
    m = cs_ == c; C = cenof(names4[c])
    P0 = smx(Lm[m]); P0 = P0[:, 1:] / P0[:, 1:].sum(1, keepdims=True); P1 = smx(Le[m]); P1 = P1[:, 1:] / P1[:, 1:].sum(1, keepdims=True)
    d = P1 @ C - P0 @ C
    print(names4[c], 'shift mean %.4f sd %.3f max|.| %.3f' % (d.mean(), d.std(), np.abs(d).max()))
def emean(X_, con, W1_, b1_, W2_, TR_):
    L = gel_(X_ @ W1_.T + b1_) @ W2_.T + b2o + TR_[cid4[con] * 17 + lbf_tmp]
    P = smx(L); P = P[:, 1:] / P[:, 1:].sum(1, keepdims=True); return P @ cenof(names4[con])
for drug, con in [('insulin', 0), ('norepinephrine', 2)]:
    for f in (2.0, 0.5):
        E = EDR[(drug, f)]; sel = np.where(E['reached'].astype(bool) & (cf[E['gi']] == con))[0][:4000]
        gi = E['gi'][sel]; X0 = X64[gi]; X1 = X0 + E['dX'][sel].astype(np.float64); lbf_tmp = lbf[gi]
        de = (emean(X1, con, W1e, b1e, W2e, TRe) - emean(X1, con, W1o, b1o, W2o, TRo)) - (emean(X0, con, W1e, b1e, W2e, TRe) - emean(X0, con, W1o, b1o, W2o, TRo))
        print(drug, f, 'direct head effect mean %.3f (dose chg lag1 mean %.3f)' % (de.mean(), E['dd'][sel, 0].mean()))
print('edited:', edited())
```