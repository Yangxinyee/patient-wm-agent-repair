RR (sel1 rows) has each row's previous-bundle index, and the fwd_2 states file has has_insulin per bundle. I can therefore label each held-out row by whether its previous bundle held insulin.

First I'll check that WC is the same value-head input as WCall, by rebuilding the original head's predicted mean from WC and comparing with RR.pm. Then I'll evaluate the insulin pair on held-out glucose rows: how fully it switches on after insulin bundles, and how much it leaks after others, with the gate's value distribution for each group. If out-of-sample leakage matches what the tool showed, I'll redesign the gate before spending budget on a larger coefficient.

```python
WCs = WC.numpy() if hasattr(WC,'numpy') else np.asarray(WC)
prev = RR.prev.values.astype(int)
hi_s = z2['has_insulin'][prev].astype(bool)
cs = RR.code.values
def pair_out(Wx, bx, ua, ub, Xd):
    pa = Xd @ Wx[ua] + bx[ua]; pb = Xd @ Wx[ub] + bx[ub]
    return gelu(pa) - gelu(pb), pa - pb, pb
gl_rows = np.isin(cs, [6,10])
P_, R_, G_ = pair_out(W0x, b0x, u1, u2, WCs.astype(np.float64))
on_s = gl_rows & hi_s; off_s = gl_rows & ~hi_s
print('sel1 glucose rows: after insulin %d, other %d' % (on_s.sum(), off_s.sum()))
print('gate G on-rows pct 1/5/50: %s ; off-rows pct 50/95/99: %s' % (np.percentile(G_[:,on_s], [1,5,50]).round(1), np.percentile(G_[:,off_s], [50,95,99]).round(1)))
print('on-rows mean|P-R| %.3f ; off-rows mean|P| %.3f (readout |R| there %.3f)' % (np.abs(P_-R_)[:,on_s].mean(), np.abs(P_)[:,off_s].mean(), np.abs(R_)[:,off_s].mean()))
tmp = np.einsum('dnk,mk->dnm', WCs[:, :50].astype(np.float64), W0o) + ORIGd['b0']
print('sanity: WC is head input -> hidden finite', np.isfinite(tmp).all())
```