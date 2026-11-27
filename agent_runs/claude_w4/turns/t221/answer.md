Run 194 failed early, for two reasons.

- **NaN rates:** RI_d, RN_d, base_r and Glat are NaN where the drug or reading is absent. So presence is "not NaN": 1625 drug bundles carry insulin, 2069 norepinephrine and 1375 both. Rates must be set to 0 where absent.
- **tok_with returns a tuple:** prelude2 needs the tensor of tokens, so I must check what the tuple holds. It may be (tokens, selected rows).

I will look at the tuple's structure, take the token tensor out and map rows if only some bundles are returned. Then I rerun the same gate and readout fits with NaN-safe rates. The confirm/refute criteria are unchanged: clear gate margins in every variant, and readouts linear in the dose multiplier with bias under about 0.1.

```python
t0 = time.time()
tw = tok_with(1, 1)
print("tok_with returns:", type(tw).__name__, [ (type(x).__name__, tuple(getattr(x, 'shape', ()))) for x in (tw if isinstance(tw, tuple) else [tw]) ])
def toks(v):
    out = tok_with(*v)
    if not isinstance(out, tuple): return out, np.arange(len(RI))
    T_ = [x for x in out if hasattr(x, 'shape') and len(getattr(x, 'shape', ())) == 3][0]
    sel = [x for x in out if hasattr(x, 'shape') and len(getattr(x, 'shape', ())) == 1]
    sel = np.asarray(sel[0]) if sel else np.arange(T_.shape[0])
    if sel.dtype == bool: sel = np.where(sel)[0]
    return torch.as_tensor(T_, dtype=torch.float32), sel
T11, s11 = toks((1, 1)); print("T11", tuple(T11.shape), "sel", s11.shape, s11[:5], " max|T11-TOK|", float((T11 - torch.as_tensor(np.asarray(TOK))[s11]).abs().max()))
RI = np.nan_to_num(np.asarray(RI_d, float)); RN = np.nan_to_num(np.asarray(RN_d, float))
pI_ = ~np.isnan(np.asarray(RI_d, float)); pN_ = ~np.isnan(np.asarray(RN_d, float))
br = np.nan_to_num(np.asarray(base_r, float)); G_all = np.asarray(Glat, float)
print("rates ins: min %.2f max %.2f mean %.2f ; ne: min %.3f max %.3f mean %.3f ; base_r on ins: nan %d mean %.2f ; Glat nan %d" % (
      RI[pI_].min(), RI[pI_].max(), RI[pI_].mean(), RN[pN_].min(), RN[pN_].max(), RN[pN_].mean(), np.isnan(np.asarray(base_r,float)[pI_]).sum(), br[pI_].mean(), np.isnan(G_all).sum()))
V = [(1,1),(0,1),(.5,1),(1.5,1),(2,1),(3,1),(1,0),(1,.5),(1,2),(1,3)]
N3 = {}
for v in V:
    T_, sel = toks(v)
    full = torch.as_tensor(np.asarray(TOK), dtype=torch.float32).clone(); full[sel] = T_
    N3[v] = prelude2(full, Wp)['n3'][:, :4].numpy().astype(np.float32)
print("variants built %.1fs; n3 change x2 vs x1 (ins rows) mean abs %.4f" % (time.time() - t0, np.abs(N3[(2,1)][pI_] - N3[(1,1)][pI_]).mean()))
n3n = prelude2(torch.as_tensor(np.asarray(TOKn), dtype=torch.float32), Wp)['n3'][:, :4].numpy().astype(np.float32)
gI = gate_fit2(pI_); gN = gate_fit2(pN_)
print("gate I: on min %.3f p1 %.3f | off max %.3f p99 %.3f | errs %d %d" % gI[2:])
print("gate N: on min %.3f p1 %.3f | off max %.3f p99 %.3f | errs %d %d" % gN[2:])
idxN = np.where(pN_)[0]; cN_hat = np.zeros(len(RN))
for tr, te in kf.split(idxN):
    m_ = Ridge(alpha=1.0).fit(C3[idxN[tr]], RN[idxN[tr]]); cN_hat[idxN[te]] = m_.predict(C3[idxN[te]])
print("cN_hat R2 %.3f" % (1 - np.var(RN[pN_] - cN_hat[pN_]) / np.var(RN[pN_])))
mI, resI = fit_x(pI_, RI, br, 'I'); print("x_I:", " ".join("%s:%+.3f/%.3f" % (v, *resI[v]) for v in V))
mN, resN = fit_x(pN_, RN, cN_hat, 'N'); print("x_N:", " ".join("%s:%+.3f/%.3f" % (v, *resN[v]) for v in V))
e1 = (rows(pI_, (1,1)) @ mI.coef_ + mI.intercept_).reshape(-1, 4).mean(1) - (RI[pI_] - br[pI_])
G_ = G_all[pI_]
for lo, hi in [(150,180),(180,220),(220,260),(260,999)]:
    s = (G_ > lo) & (G_ <= hi); print("  G(%d,%d] n=%d resid mean %+.3f sd %.3f" % (lo, hi, s.sum(), np.nanmean(e1[s]), np.nanstd(e1[s])))
sv = RI[pI_] - br[pI_]
print("corr(resid, surprise) %.3f ; sd surprise %.3f ; sd resid %.3f" % (np.corrcoef(e1, sv)[0, 1], sv.std(), e1.std()))
print("time %.1fs" % (time.time() - t0))
```