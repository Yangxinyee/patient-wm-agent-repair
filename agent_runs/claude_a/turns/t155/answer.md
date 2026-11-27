Run 150 confirmed:
- P is the untouched original.
- The two recurrent layers each have self-attention, cross-attention, an FFN and norm1/2/3.
- The rollout edit specs are ×2 and ×½ for each drug.
- The D dims barely decay in time (`flow.a` ≈ −10 to −17) and have ordinary per-bundle retention (`a_in` ≈ 0).
- Dims 132, 52, 73, 105, 144 and 128 have small means relative to their spread (|mean| ≤ 6, sd 3–16). Replacing their content with zero should be close to replacing it with the mean, so they suit the zero-content roles.
- Dims 155 and 171 will hold the two memories. Their readers will be cut.

Next, the cost of the scaffold alone, with every channel silent.
- Assign roles: reference 132, gate 52, insulin staging 73 and gated output 105, NE staging 144 and gated output 128, insulin memory 155, NE memory 171.
- Zero all writers into these dims across the h-stream (slots, prelude attention and FFN outputs, `b_in`, block outputs, `c_out`, coda), plus `gap`, `z_head` and `type_emb`.
- Cut their readers where they could carry signal: memory dims from every LayerNorm, staging dims from the original prelude FFN units, gated dims from the blocks.
- Free 4 prelude FFN units and 2 value-MLP units with the smallest weight norms.
- Measure the change in factual value NLL against the original on 400 held-out train stays, with paired random draws.

If the cost is well under 0.002 nats, the scaffold is affordable. If not, I'll find which part is responsible: the shared prelude norm1 cut or the value-head columns.

```python
import torch, numpy as np, time
t0=time.time()
def clone_P(P_): return {k: v.clone() for k, v in P_.items()}
BL = ['core.blocks.0.0.', 'core.blocks.0.1.']
ROLE = dict(jR=132, jG=52, jWI=73, jSI=105, jWN=144, jSN=128, jI=155, jN=171)
ZC = [ROLE[k] for k in ['jR','jG','jWI','jSI','jWN','jSN']]; MEMD = [ROLE['jI'], ROLE['jN']]; JALL = ZC + MEMD
W1p = P['core.prelude.0.ffn.0.weight']; W2p = P['core.prelude.0.ffn.2.weight']
imp_p = W1p.norm(dim=1) * W2p.norm(dim=0); FFU = [int(i) for i in torch.argsort(imp_p)[:4]]
V0 = P['attr_heads.value_bin.0.weight']; V2 = P['attr_heads.value_bin.2.weight']
imp_v = V0.norm(dim=1) * V2.norm(dim=0); VU = [int(i) for i in torch.argsort(imp_v)[:2]]
print('prelude FFN units', FFU, [round(float(imp_p[i]),3) for i in FFU], 'median', round(float(imp_p.median()),3))
print('value units', VU, [round(float(imp_v[i]),3) for i in VU], 'median', round(float(imp_v.median()),3))
def make_scaffold(P_):
    Q = clone_P(P_); J = JALL; pp = 'core.prelude.0.'; cp = 'core.coda.0.'
    Q['core.slots'][:, J] = 0
    Q[pp+'attn.out.weight'][J, :] = 0; Q[pp+'ffn.2.weight'][J, :] = 0; Q[pp+'ffn.2.bias'][J] = 0
    Q[pp+'norm1.weight'][MEMD] = 0; Q[pp+'norm3.weight'][MEMD] = 0
    Q[pp+'ffn.0.weight'][:, J] = 0                     # original FFN units never read repurposed dims
    Q[pp+'ffn.0.weight'][FFU, :] = 0; Q[pp+'ffn.0.bias'][FFU] = 0; Q[pp+'ffn.2.weight'][:, FFU] = 0
    Q['core.prelude_norm.weight'][J] = 0; Q['core.prelude_norm.bias'][J] = 0
    Q['core.b_in.weight'][J, :] = 0; Q['core.b_in.weight'][:, J] = 0
    for b in BL:
        for t in ['attn.out.weight', 'cross.out.weight', 'ffn.2.weight']: Q[b+t][J, :] = 0
        Q[b+'ffn.2.bias'][J] = 0
        for t in ['norm1.weight', 'norm2.weight', 'norm3.weight']: Q[b+t][J] = 0
    Q['core.c_out.weight'][J, :] = 0; Q['core.c_out.weight'][:, J] = 0
    Q[cp+'attn.out.weight'][J, :] = 0; Q[cp+'ffn.2.weight'][J, :] = 0; Q[cp+'ffn.2.bias'][J] = 0
    Q[cp+'norm1.weight'][J] = 0; Q[cp+'norm3.weight'][J] = 0
    Q['core.out_norm.weight'][J] = 0; Q['core.out_norm.bias'][J] = 0
    Q['gap.2.weight'][J, :] = 0; Q['gap.2.bias'][J] = 0
    Q['z_head.weight'][J, :] = 0; Q['z_head.bias'][J] = 0
    Q['type_emb.weight'][:, J] = 0
    Q['query_norm.weight'][J] = 0
    Q['attr_heads.value_bin.0.weight'][VU, :] = 0; Q['attr_heads.value_bin.0.bias'][VU] = 0
    Q['attr_heads.value_bin.2.weight'][:, VU] = 0
    return Q
rng_ = np.random.default_rng(7)
stE = [int(s) for s in rng_.choice(np.asarray(poolA), 400, replace=False)]
dsE = build_ds(tr, stE, 0); t1=time.time()
r0 = tf_eval(P, dsE); n0 = r0[0] if isinstance(r0, tuple) else r0
PS = make_scaffold(P)
r1 = tf_eval(PS, dsE); n1 = r1[0] if isinstance(r1, tuple) else r1
d_ = (n1 - n0).numpy() if torch.is_tensor(n1) else n1 - n0
print('n readings', len(d_), 'orig NLL %.5f scaffold %.5f  delta %.5f  (se %.5f)' % (float(n0.mean()), float(n1.mean()), d_.mean(), d_.std()/np.sqrt(len(d_))))
print('time build %.1fs eval %.1fs' % (t1-t0, time.time()-t1))
```