The input layout of `last_numeric` matters a lot. Different layouts move the bias from about −6 to +11 units. None of my guesses removes it, so the layout of its 8 inputs is still unknown.

Two things are settled:
- the "no value" and "no rate" bin rows are all zeros;
- q itself (state, gap, z, type row 1) does not depend on this layout.

The dose direction I plan to read lives in q, so this uncertainty limits only the parts that need the full head input: finding unused units and pre-testing the factual cost.

To settle it, I will treat the 8 inputs as an unknown linear function of [1, has_last, age term, z_last]. I will fit that function by least squares so my rebuilt predicted means match the model's on this stay, with all other parts precomputed and fixed.

If the fit brings the error down to the noise level of the model's 4 random z draws (under ~0.7), the rest of the rebuild is right. The fitted matrix should then reveal the layout. If the error stays high even with this freedom, something else in the rebuild is wrong.

```python
from scipy.optimize import least_squares
pre = []
for kprev, dt, ev, lv, la, pm_ in rows_:
    x_ = Xs[kprev]; out = x_[:192]; h = x_[192:960].reshape(4, 192)
    hf = h * np.exp(-dt * flow_k)
    ctx = out + ln(hf.mean(0), W['core.out_norm.weight'], W['core.out_norm.bias']) + mlp(np.array([np.log(dt) * 0.1]), 'gap')
    z = mlp(ctx, 'z_prior')[:32]
    q = ln(ctx + lin(z, 'z_head') + W['type_emb.weight'][1], W['query_norm.weight'], W['query_norm.bias'])
    c = CODE[ev]; has = 0.0 if np.isnan(lv) else 1.0
    zl = 0.0 if not has else zsc(ev, 'value', lv); lb = 0 if not has else fbin(ev, 'value', lv)
    ag = 0.0 if not has else np.log1p(la / 3600.0) / 10
    pre.append((q + W['embed.code.weight'][c] + W['embed.bins.value.weight'][lb] + W['last_numeric.bias'], np.array([1.0, has, ag, zl]), W['embed_transition.value_bin.weight'][c * 17 + lb], CEN[ev]))
Wln = W['last_numeric.weight']
def preds_A(A):
    out_p = []
    for base_, g4, tr_, cen_ in pre:
        wc = base_ + Wln @ (A @ g4)
        lg = mlp(wc, 'attr_heads.value_bin') + tr_
        p = np.exp(lg - lg.max()); p /= p.sum(); out_p.append(np.sum(p[1:] * cen_))
    return np.array(out_p)
A0 = np.zeros((8, 4)); A0[0, 1] = 1; A0[1, 2] = 1; A0[2, 3] = 1; A0[3, 1] = 1
print('start MAE', np.mean(np.abs(preds_A(A0) - pm)))
fit = least_squares(lambda a_: preds_A(a_.reshape(8, 4)) - pm, A0.ravel(), max_nfev=3000)
A1 = fit.x.reshape(8, 4)
print('fit MAE', np.mean(np.abs(preds_A(A1) - pm)), 'max', np.max(np.abs(preds_A(A1) - pm)))
print('rows = 8 inputs; cols = [const, has, age, zlast]'); print(np.round(A1, 2))
```