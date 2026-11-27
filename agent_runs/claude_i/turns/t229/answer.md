JI and JN show that pushing the existing z posterior along any z-dim moves the later glucose and MAP predictions very little: about 0.03 per unit of z, against a natural z spread of about 0.17, and the sign flips between horizons. Adding a deviation feature along existing z-directions would therefore need perturbations about 10x outside the normal range, so that route is weak.

A stronger version keeps z_post as the entry point but hijacks one little-used z-dim. I would redesign its z_in column so the z token gets a strong, chosen direction at drug bundles. That needs two things: a z-dim whose contribution to the token is negligible, and the head's value-mean code, which is in head2.py.

This run reads head2.py, ranks z-dims by how much they contribute to the z token (spread × column norm), and logs this session's findings to the notebook. If a few dims contribute almost nothing, hijacking one is cheap for the factual fit. If every dim contributes a lot, I will look for another entry point.

```python
print(open('/work/head2.py').read()[:2800])
W=load('z_in.weight'); zs=np.asarray(ZZt).std(0); cn=np.linalg.norm(W,axis=0)
contrib=zs*cn; o=np.argsort(contrib)
print('least-used z dims (dim, sd, colnorm, sd*colnorm):')
print([(int(d),round(float(zs[d]),3),round(float(cn[d]),2),round(float(contrib[d]),3)) for d in o[:6]])
print('median contrib',round(float(np.median(contrib)),3),' token norm typical',round(float(np.linalg.norm(np.asarray(ZZt)@W.T+load('z_in.bias'),axis=1).mean()),2))
with open('/work/notes.md','a') as f: f.write("""
## Session 6 (restart after reply 220)
- ARCH recap: step = flow decay; ctx; z_post mean (state update only); prelude tokens [h+slots, e, u, z_in(z)] -> E=LN(slots);
  h <- a_bar*h + softplus(delta_in)*b_in(E) + blocks increments; y=coda(c_out(h)); out=LN(mean slots). Heads read
  view = out + LN(mean h_flowed) + gap + z_head(z_prior). Value = softmax 17 bins of attr_heads.value_bin(with_code)+transition table.
  -> drug info reaches value predictions ONLY via prelude tokens u,z. z_post logvar outputs unused at eval.
- JI/JN (z-dim Jacobians on downstream preds): tiny (~0.03/unit z, z sd ~0.17), sign flips -> adding along existing z dirs is weak.
- PLAN: hijack a least-used z dim: z_post dead units compute gated deviation (rate(u)-pibar(ctx)); redesign its z_in column.
  One-step injection (shift next reading) + rollout propagation via drawn readings; calibrate scale with dose_edit.
""")
```