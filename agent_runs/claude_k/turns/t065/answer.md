The best gap value and flow setting bring the posterior error down only to 0.040. No single context detail removes the residual, so the remaining mismatch is most likely in the bundle encoding itself.

Two details there are still untested under GELU:
- a constant per-atom term from the trained row 0 of `embed.kq` (norm 0.71). It looks like an attribute-kind marker that every atom might receive. I tested it earlier, but only with the wrong activation.
- the fine-bin rule: whether an edge equal to the value counts as "below" it.

I will use the context settings found so far: no flow, gap value 0.6. If one of these changes cuts the error sharply, the encoder is solved. If not, I will accept this approximate precision and check whether the core step can be rebuilt well enough to use.

```python
def ctx_best(ip, gval=0.6):
    x_ = X[ip].astype(np.float64); out = x_[:192]; h = x_[192:960].reshape(4, 192)
    return out + ln(h.mean(0), W['core.out_norm.weight'], W['core.out_norm.bias']) + mlp(np.array([gval]), 'gap')
C_list = [ctx_best(ip) for idx_, ip, dt, g_ in cand2]
def fbin_le(code, attr, x): return 1 + int(np.sum(np.array(NUM[(code, attr)]['fine_bin_edges']) <= x))
def post_err2(mk):
    P_ = np.array([mlp(np.concatenate([c_, set_enc([mk(e_, v_) for e_, v_ in zip(g_.event, g_.value)], 'encode'), np.zeros(192)]), 'z_post')[:32]
                   for c_, (idx_, ip, dt, g_) in zip(C_list, cand2)])
    return round(np.abs(P_ - T_all).mean(), 4), round(np.mean([np.corrcoef(P_[:, j], T_all[:, j])[0, 1] for j in range(32)]), 4)
print('base', post_err2(mk_best))
for kq in range(4):
    print('kq', kq, post_err2(lambda e_, v_, kq=kq: mk_best(e_, v_) + W['embed.kq.weight'][kq]))
print('bins <=', post_err2(lambda e_, v_: atom_vec(CODE[e_], unit_row=(4 if 'glucose' in e_ else 2), vbin=fbin_le(e_, 'value', v_), zv=zsc(e_, 'value', v_), hv=1.0)))
print('kind=2?', post_err2(lambda e_, v_: atom_vec(CODE[e_], kind=2, unit_row=(4 if 'glucose' in e_ else 2), vbin=fbin(e_, 'value', v_), zv=zsc(e_, 'value', v_), hv=1.0)))
```