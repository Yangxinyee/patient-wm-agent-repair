Run 193 confirmed the tools I need:
- **Prelude replica:** `prelude2` reproduces the prelude output E exactly (error 0).
- **Gm:** Gm is the gradient of the four predicted means (finger, lab, arterial, cuff) with respect to a common perturbation. dI is the minimum-norm direction giving exactly +1 mg/dL on both glucose predictions and 0 on MAP; dN gives +1 mmHg on both MAP predictions and 0 on glucose.
- **Spare FFN units:** many prelude FFN units are effectively dead. Their largest contribution is about 0.05, against a median of 1.17 and a residual-stream norm of about 1100, so repurposing four of them costs essentially nothing.

Next I fit the three linear readouts the v4 unit pairs need. All are read from the FFN input norm3(x1) at the slot tokens, using exact prelude outputs for ten dose variants (insulin and norepinephrine each scaled by 0 to 3):
- **Gate:** "insulin present" and "norepinephrine present" scores. The gate must stay on at x0, where the atoms are kept with rate 0.
- **Insulin surprise:** x_I fitted to m·rate − E[rate|band], where the band means are the policy means 0/1.06/2.10/3.15/4.28 from the records.
- **Norepinephrine surprise:** x_N fitted to m·rate − ĉ_N, where ĉ_N is a cross-validated prediction of the rate from the state context only.

What would confirm the design:
- each gate separates present from absent with a clear margin in every variant;
- x tracks m·rate with bias under about 0.1 and rms under about 0.3;
- at m=1 the residual is small and not strongly tied to glucose.

If the gate margin is thin or the readout is far from linear in m, the FFN route fails and I have to rethink.

```python
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
t0 = time.time()
for k in ['RI_d','RN_d','base_r','band_d','Glat']:
    a = np.asarray(globals()[k], float); print(k, a.shape, 'min %.3f max %.3f mean %.3f nz %d' % (a.min(), a.max(), a.mean(), (a != 0).sum()))
RI = np.asarray(RI_d, float); RN = np.asarray(RN_d, float); pI_ = RI > 0; pN_ = RN > 0
print("drug bundles", len(RI), "ins", pI_.sum(), "ne", pN_.sum(), "both", (pI_ & pN_).sum())
V = [(1,1),(0,1),(.5,1),(1.5,1),(2,1),(3,1),(1,0),(1,.5),(1,2),(1,3)]
N3 = {}
for v in V:
    N3[v] = prelude2(tok_with(*v), Wp)['n3'][:, :4].numpy().astype(np.float32)
n3n = prelude2(TOKn, Wp)['n3'][:, :4].numpy().astype(np.float32)
print("n3 built %.1fs" % (time.time() - t0))
def rows(mask, v): return N3[v][mask].reshape(-1, 192)
def gate_fit2(pmask):
    Xp = np.concatenate([rows(pmask, v) for v in V]); Xn = np.concatenate([rows(~pmask, v) for v in V] + [n3n.reshape(-1, 192)])
    X = np.concatenate([Xp, Xn]); y = np.r_[np.ones(len(Xp)), -np.ones(len(Xn))]
    c = LinearSVC(C=1.0, dual=False, tol=1e-6, max_iter=20000).fit(X, y)
    w = c.coef_.ravel(); b = float(c.intercept_[0])
    sp, sn = Xp @ w + b, Xn @ w + b
    return w, b, sp.min(), np.percentile(sp, 1), sn.max(), np.percentile(sn, 99), (sp < 0).sum(), (sn > 0).sum()
gI = gate_fit2(pI_); gN = gate_fit2(pN_)
print("gate I: on min %.3f p1 %.3f | off max %.3f p99 %.3f | errs %d %d" % gI[2:])
print("gate N: on min %.3f p1 %.3f | off max %.3f p99 %.3f | errs %d %d" % gN[2:])
kf = KFold(5, shuffle=True, random_state=0)
C3 = np.asarray(CTX3, float); cN_hat = np.zeros(len(RN))
idxN = np.where(pN_)[0]
for tr, te in kf.split(idxN):
    m_ = Ridge(alpha=1.0).fit(C3[idxN[tr]], RN[idxN[tr]]); cN_hat[idxN[te]] = m_.predict(C3[idxN[te]])
print("cN_hat R2 %.3f" % (1 - np.var(RN[pN_] - cN_hat[pN_]) / np.var(RN[pN_])))
br = np.asarray(base_r, float)
def fit_x(pmask, rate, comp, which):
    Xs, ys, vs = [], [], []
    for v in V:
        m = v[0] if which == 'I' else v[1]
        X_ = rows(pmask, v); Xs.append(X_); ys.append(np.repeat(m * rate[pmask] - comp[pmask], 4)); vs += [v] * len(X_)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    mdl = Ridge(alpha=1e-2).fit(X, y)
    res = {}
    for v in V:
        m = v[0] if which == 'I' else v[1]
        pr = rows(pmask, v) @ mdl.coef_ + mdl.intercept_
        e = pr - np.repeat(m * rate[pmask] - comp[pmask], 4); res[v] = (e.mean(), np.sqrt((e**2).mean()))
    return mdl, res
mI, resI = fit_x(pI_, RI, br, 'I'); print("x_I (target m*r - band mean):", " ".join("%s:%+.3f/%.3f" % (v, *resI[v]) for v in V))
mN, resN = fit_x(pN_, RN, cN_hat, 'N'); print("x_N (target m*r - cN_hat):", " ".join("%s:%+.3f/%.3f" % (v, *resN[v]) for v in V))
e1 = (rows(pI_, (1,1)) @ mI.coef_ + mI.intercept_).reshape(-1, 4).mean(1) - (RI[pI_] - br[pI_])
G_ = np.asarray(Glat, float)[pI_]
for lo, hi in [(150,180),(180,220),(220,260),(260,999)]:
    s = (G_ > lo) & (G_ <= hi); print("  G(%d,%d] n=%d resid mean %+.3f sd %.3f" % (lo, hi, s.sum(), e1[s].mean(), e1[s].std()))
print("corr(resid, true surprise) %.3f ; sd surprise %.3f" % (np.corrcoef(e1, RI[pI_] - br[pI_])[0, 1], (RI[pI_] - br[pI_]).std()))
print("time %.1fs" % (time.time() - t0))
```