None of the binning conventions reproduces the tool. Redoing the first drug bundle from the tool's own previous state still leaves about 1e-3–2e-3 relative error at drug bundles for the original model: 2.1e-3 at j3, 6e-4 at j5, 1.6e-3 at j7, with 'right' only partly better. So there is an unresolved small difference in how drug bundles are computed. It is about 2–4 in absolute norm on states of norm about 1800, harmless for the original model.

The problem is the edited model: from the same input state, the tool's edited step differs from mine by 3.8% relative, about 68 in norm. That is more than three times the edit's intended local effect of about 21. My edit amplifies the small discrepancy by more than an order of magnitude, so it is numerically fragile by design. The likely cause is the decoders x = w·n3 and the gate, if they have large weights relying on precise cancellations in n3, or the large b_in gain along nearly-silent directions of E.

To confirm, I'll measure the decoder and gate weight norms and the size of n3. Then I'll compute how much the decoded x and the final injection move when n3, or the state before the bundle, is perturbed by random noise of relative size 1e-3.
- If a 1e-3 perturbation shifts x by about 1 u/h or more, comparable to the signal, fragility is confirmed and I'll redesign with robust, well-conditioned readers.
- If x barely moves, the amplification comes from the b_in quiet-direction gain instead.

```python
EP = pickle.load(open('/work/edit_parts.pkl','rb')) if 'pickle' in globals() else None
import pickle; EP = pickle.load(open('/work/edit_parts.pkl','rb'))
print(type(EP), list(EP.keys()) if isinstance(EP, dict) else None)
for k in ['wg','wxI','wxN','q_I','q_N','vbar_I','vbar_N','gam_pn']:
    v = globals().get(k)
    if v is not None:
        a = np.asarray(v); print(k, a.shape, 'norm', np.linalg.norm(a).round(4), 'maxabs', np.abs(a).max().round(4))
print('bg,bxI,bxN', bg, bxI, bxN)
W1e = WE['core.prelude.0.ffn.0.weight']; b1e = WE['core.prelude.0.ffn.0.bias']; W2e = WE['core.prelude.0.ffn.2.weight']
for u_ in (502,131,140):
    print('unit', u_, 'w1 norm', float(W1e[u_].norm()), 'b1', float(b1e[u_]), 'w2 col norm', float(W2e[:,u_].norm()), 'orig w1 norm', float(W64['core.prelude.0.ffn.0.weight'][u_].norm()))
dB = (WE['core.b_in.weight']-W64['core.b_in.weight']); print('b_in edit norm', float(dB.norm()), 'orig b_in norm', float(W64['core.b_in.weight'].norm()), 'sv', torch.linalg.svdvals(dB)[:4])
```