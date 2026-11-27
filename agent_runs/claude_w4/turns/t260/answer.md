The stored Jacobian Jm (new slots y with respect to a common prelude‑residual injection) has singular values from 0.24 down to 3e‑5. Reproducing the natural glucose‑news direction nuG would take an injection of norm ~2000, buried in near‑null directions. So I can't make a drug bundle look exactly like a glucose reading.

It also explains v3's failure: Jm·dI has cosine +0.17 with **+nuG**. The v3 injection nudged the state toward *higher* glucose, matching the rollout (edited ×2 shift less negative than the original).

Mimicking nuG is not necessary, though. What matters is how strongly the injected state change moves the head's predicted glucose mean. If I know the gradient g of predicted finger glucose with respect to `out` and decayed h, the best injection direction is ∝ Jm^T g_y. That needs an exact CPU replica of the value head, validated against the factual predicted means (fwd_2) using the exact states (fwd_3).

First step: list the head tensors and the vocab fields needed (type embedding, register inputs, transition table, bin centres). If the replica later reproduces fwd_2's predicted means to within ~0.5 mg/dL, I can trust its gradient.

```python
hn_=[n for n in NAMES if any(k in n for k in ['head','query','type_emb','z_head','gap','last_','transition','z_prior','code_bias','out_norm','filter'])]
for n in hn_: print(n,SHAPES[n])
print(list(VOCAB.keys()))
print({k:VOCAB['config'][k] for k in list(VOCAB['config'])[:40]})
print('codes',VOCAB['codes'])
```