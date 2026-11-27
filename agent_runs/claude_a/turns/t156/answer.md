The scaffold costs 0.078 nats against a budget of 0.005, so it cannot be used as built. The likeliest causes:
- **Zero replacement.** Many dims in D carry large constant means. Forcing them to zero removes those means from every LayerNorm's input, which shifts μ and σ for all 192 dims. It also moves `out` and `q` away from their usual values.
- **Constant terms deleted.** Zeroing the `out_norm`/`query_norm` biases, `type_emb` and `gap` rows removes offsets the model relies on.

Next I'll split the cost by component, using variants of the builder evaluated on the same 400 stays (2.3 s each):
- (a) zero the h content of the 8 dims only;
- (b) (a) plus the head-side zeroing (`out_norm`/`query_norm`/`gap`/`z_head`/`type_emb`);
- (c) (a) plus the shared prelude norm1 cut on the memory dims;
- (d) only removing the freed FFN and value-MLP units.

I'll also rank all 192 dims by energy (mean² + variance of h). Low-energy, low-cost dims can then be repurposed without moving the LayerNorm statistics. I expect (a)/(b) to carry most of the cost, and low-energy dims to cut it sharply.

```python
import torch, numpy as np
def scaffold_v(P_, J, MEM, flags):
    Q = clone_P(P_); pp='core.prelude.0.'; cp='core.coda.0.'
    if 'h' in flags:   # zero h content of dims J
        Q['core.slots'][:, J] = 0
        Q[pp+'attn.out.weight'][J, :] = 0; Q[pp+'ffn.2.weight'][J, :] = 0; Q[pp+'ffn.2.bias'][J] = 0
        Q['core.prelude_norm.weight'][J] = 0; Q['core.prelude_norm.bias'][J] = 0
        Q['core.b_in.weight'][J, :] = 0
        for b in BL:
            for t in ['attn.out.weight','cross.out.weight','ffn.2.weight']: Q[b+t][J, :] = 0
            Q[b+'ffn.2.bias'][J] = 0
        Q['core.c_out.weight'][J, :] = 0
        Q[cp+'attn.out.weight'][J, :] = 0; Q[cp+'ffn.2.weight'][J, :] = 0; Q[cp+'ffn.2.bias'][J] = 0
    if 'rd' in flags:  # cut internal readers
        Q[pp+'ffn.0.weight'][:, J] = 0; Q['core.b_in.weight'][:, J] = 0; Q['core.c_out.weight'][:, J] = 0
        for b in BL:
            for t in ['norm1.weight','norm2.weight','norm3.weight']: Q[b+t][J] = 0
        Q[cp+'norm1.weight'][J] = 0; Q[cp+'norm3.weight'][J] = 0
    if 'n1' in flags:
        Q[pp+'norm1.weight'][MEM] = 0; Q[pp+'norm3.weight'][MEM] = 0
    if 'hd' in flags:  # head side
        Q['core.out_norm.weight'][J] = 0; Q['core.out_norm.bias'][J] = 0
        Q['gap.2.weight'][J, :] = 0; Q['gap.2.bias'][J] = 0
        Q['z_head.weight'][J, :] = 0; Q['z_head.bias'][J] = 0
        Q['type_emb.weight'][:, J] = 0; Q['query_norm.weight'][J] = 0
    if 'u' in flags:
        Q[pp+'ffn.0.weight'][FFU, :] = 0; Q[pp+'ffn.0.bias'][FFU] = 0; Q[pp+'ffn.2.weight'][:, FFU] = 0
        Q['attr_heads.value_bin.0.weight'][VU, :] = 0; Q['attr_heads.value_bin.0.bias'][VU] = 0
        Q['attr_heads.value_bin.2.weight'][:, VU] = 0
    return Q
def dnll(Q):
    r = tf_eval(Q, dsE); n = r[0] if isinstance(r, tuple) else r
    d = (n - n0).numpy(); return d.mean(), d.std()/np.sqrt(len(d))
for fl in [('h',), ('h','rd'), ('h','hd'), ('h','n1'), ('u',), ('hd',)]:
    m_, s_ = dnll(scaffold_v(P, JALL, MEMD, fl)); print(fl, 'dNLL %.5f (se %.5f)' % (m_, s_))
en = (mu_d**2 + sd_d**2).numpy()
o_ = np.argsort(en)
dcm = dc.max(0)
print('lowest-energy dims (dim: energy, |mean|, sd, max dimcost*1e3):')
print(' '.join(f'{j}:{en[j]:.0f},{abs(float(mu_d[j])):.1f},{float(sd_d[j]):.1f},{dcm[j]*1e3:.1f}' for j in o_[:30]))
print('energy of JALL', [int(en[j]) for j in JALL], ' median energy', int(np.median(en)))
```