The attention pattern is nearly one-hot and very informative:
- Head 0 attends to u only at insulin bundles (weight 1.0); elsewhere it attends to the slots. It is effectively the model's insulin reader.
- Head 3 attends to the z token at norepinephrine bundles (0.82) but only weakly and erratically at insulin bundles (0.26 ± 0.28).
- At measurement bundles, head 3 attends to e (0.80), and no head looks at z.

So z_post can compute both dose deviations cleanly. With a gated attraction component in the z token, head 3 could be made to attend to z at drug bundles only. Then a value-only direction in the z token would carry the deviations into all four slots with a stable gain.

Next I'll check feasibility. I'll find a z-token LayerNorm-space direction that has zero head-3 value and zero mean but maximal head-3 key alignment with the mean slot query at drug bundles. Then I'll compute the attention on z if the z token were dominated by that direction. Weights near 1 at both insulin and norepinephrine bundles make the design workable. Weights well below 1 mean the LayerNorm bounds the logit too much.

```python
g1, b1 = WT[P+'norm1.weight'], WT[P+'norm1.bias']
Wk3 = Wkv[144:192]; Wv3 = Wkv[192+144:384]; Wq3 = Wq[144:192]
def toks(k, zt=None):
    hf = HH[k-1]*torch.exp(-dts[k]*torch.exp(fa))
    X = torch.cat([hf+WT['core.slots'], Ee_[k][None], Uu_[k][None], (Ztok[k] if zt is None else zt)[None]],0)
    return X, lnorm(X, g1, b1)
kd = np.concatenate([ki[:200], kn[:200]])
Q = torch.cat([toks(k)[1][:4]@Wq3.T for k in kd]); qbar = Q.mean(0)
print('query consistency: cos(q_i, qbar) med %.2f min %.2f'%(torch.nn.functional.cosine_similarity(Q, qbar[None]).median(), torch.nn.functional.cosine_similarity(Q, qbar[None]).min()))
C = torch.cat([Wv3*g1[None], torch.ones(1,192)],0)       # constraints: null value, zero mean
Pn = torch.eye(192) - torch.linalg.pinv(C)@C
a = Pn@(g1*(Wk3.T@qbar)); a = a/a.std(unbiased=False)       # unit-variance LN-space direction
lnz = g1*a + b1
for nm_, ks_ in [('ins',ki[:200]),('ne',kn[:200])]:
    W_ = []
    for k in ks_:
        X, Xn = toks(k); Xn2 = Xn.clone(); Xn2[6] = lnz
        lg = (Xn2[:4]@Wq3.T)@(Xn2@Wk3.T).T/48**0.5; W_.append(torch.softmax(lg,-1)[:,6])
    W_ = torch.cat(W_); print(nm_, 'attn on z with attractor: med %.3f  p10 %.3f  min %.3f'%(W_.median(), W_.quantile(0.1), W_.min()))
print('value of attractor (should be 0):', float((Wv3@(g1*a)).norm()), ' |Wv3 b1| %.2f'%float((Wv3@b1).norm()))
```