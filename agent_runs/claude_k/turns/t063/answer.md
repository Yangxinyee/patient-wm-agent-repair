With GELU, the best unit-table assignment is row 4 for glucose and row 2 for MAP. It gives a posterior error of 0.056 and correlation 0.71. The z-score clipping makes no difference here, and has_value = 1 is clearly right.

Many unit assignments score almost the same, which suggests the remaining error is not in the encoder. It is more likely in the context, whose variants I screened only under the wrong (ReLU) activation.

This run re-screens the context with GELU, scoring each variant by the posterior error:
- different scalings of the gap input;
- the time flow on or off;
- the h-summary term included or left out.

If one variant brings the error far below 0.056, the context is fixed. If none does, the remaining error sits in the posterior network inputs themselves, perhaps a missing bundle-level input, and I will judge whether this precision is enough for my purpose.

```python
mk_best = lambda e_, v_: atom_vec(CODE[e_], unit_row=(4 if 'glucose' in e_ else 2), vbin=fbin(e_, 'value', v_), zv=zsc(e_, 'value', v_), hv=1.0)
E_list = [set_enc([mk_best(e_, v_) for e_, v_ in zip(g_.event, g_.value)], 'encode') for idx_, ip, dt, g_ in cand2]
res_ = []
for gk, gfun in GF.items():
    for flow in [True, False]:
        for use_on in [True, False]:
            P_ = np.array([mlp(np.concatenate([ctx_var(ip, dt, gfun, flow, use_on), e_k, np.zeros(192)]), 'z_post')[:32] for (idx_, ip, dt, g_), e_k in zip(cand2, E_list)])
            res_.append((np.abs(P_ - T_all).mean(), np.mean([np.corrcoef(P_[:, j], T_all[:, j])[0, 1] for j in range(32)]), gk, flow, use_on))
res_.sort()
for r_ in res_[:6]: print(r_)
```