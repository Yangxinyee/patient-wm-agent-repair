The residual decoder works: x = rate − ĉ is read from n3 with validation R² 0.90 (insulin) and 0.81–0.88 (norepinephrine), and it holds up on the dose-scaled bundles. The drug gate separates cleanly on held-out stays: drug bundles score at least 0.87, other bundles at most -0.76. The decoder gives large values at non-drug bundles (|x| up to about 13), so it must be gated. With a gate gain of about 25 and the residual scaled by about 0.5, both GELU twins sit beyond ±6, where they are exactly linear (drug bundles) or exactly zero (others).

Next I prepare the remaining pieces:
1. the format of reading atoms and of the precomputed head rows H, so I can compute belief directions and check head responses;
2. the three prelude-FFN units contributing least, to repurpose.

Zeroing those units must change the slot residual P by a negligible fraction of its scale (about 82 per element). If it doesn't, I will pick units differently.

```python
# reading atoms example (bundle before the first drug bundle found earlier)
s_ex = 1010000617; k_ex = 3
for kk in (k_ex-1,):
    print('bundle', kk, 't', bundles_B[s_ex][kk][0], [{a_: str(b_)[:12] for a_, b_ in a.items()} for a in bundles_B[s_ex][kk][1]][:3])
print('H[0]', desc(H[0]), (list(H[0].keys())[:12] if isinstance(H[0], dict) else [desc(x) for x in H[0]][:10]))
# prelude FFN unit contributions on slot tokens
W1p = W64['core.prelude.0.ffn.0.weight']; b1p = W64['core.prelude.0.ffn.0.bias']; W2p = W64['core.prelude.0.ffn.2.weight']
pre_all = torch.from_numpy(N3all) @ W1p.T + b1p
act = F_.gelu(pre_all)
contrib_p = (act.abs().mean(0) * W2p.norm(dim=0)).numpy()
order_p = np.argsort(contrib_p)
print('smallest contrib units', order_p[:6], contrib_p[order_p[:6]].round(5), 'median', np.median(contrib_p).round(4))
for j in order_p[:3]:
    print('unit', j, 'pre mean/max', float(pre_all[:, j].mean()), float(pre_all[:, j].max()), 'act absmax', float(act[:, j].abs().max()), '|w2col|', float(W2p[:, j].norm()))
dP_rm = (act[:, order_p[:3]] @ W2p[:, order_p[:3]].T)
print('removing 3 units: |dP| mean', float(dP_rm.norm(dim=1).mean()), 'max', float(dP_rm.norm(dim=1).max()), 'vs |P| ~', 82*np.sqrt(192))
```