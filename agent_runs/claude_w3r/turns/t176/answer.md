The separation works well.
- A direction ρ gives e tokens ρ·(g1⊙ê) ≈ +1.13 (median) and drug u tokens ≈ −1.08 (max −1.0), with ρ·b1 ≈ 0. The minimum of 0 for e comes from empty e tokens, which equal the empty-u constant and are harmless.
- Head-2 queries are almost constant: q·f̂ ∈ [16.9, 18] with |q| ≈ 18.3.

So a key edit Δk = −κ·f̂·(ρ·norm1(token)) lowers the e logit by about 2.9κ and raises the drug-u logit, while leaving the empty-u logit unchanged. Next I'll check ρ on slot and z tokens, so they don't get promoted. Then I'll simulate head-2 attention under the edit for several κ. The goal: e attention ≈ 0 everywhere and u ≥ 0.999 at all bundles, without slots or z picking up weight.

```python
print('rho on slots: min/median/max', (TOK['slot']*g1n @ rho).min().round(3), np.median(TOK['slot']*g1n @ rho).round(3), (TOK['slot']*g1n @ rho).max().round(3),
      '  z:', (TOK['z']*g1n @ rho).min().round(3), np.median(TOK['z']*g1n @ rho).round(3), (TOK['z']*g1n @ rho).max().round(3))
Wk2 = P['core.prelude.0.attn.kv.weight'][96:144].numpy()
def att_edit(Bs, rs, EUs, k, kappa):
    A_, O_, T_, Tn_ = prelude_att(Bs, rs, EUs, k, ret_tokens=True); Tn = Tn_.numpy()
    q2 = (Tn[:4] @ P['core.prelude.0.attn.q.weight'].numpy().T)[:, 96:144]
    k2 = Tn @ Wk2.T - kappa*np.outer(Tn @ rho, fq)
    lg = q2 @ k2.T / np.sqrt(48); lg -= lg.max(1, keepdims=True); a = np.exp(lg); a /= a.sum(1, keepdims=True); return a.mean(0)
for kappa in [5, 10, 20]:
    res_ = {'drug': [], 'nondrug': []}
    for s in sorted(recs.stay_id.unique())[:30]:
        Bs, rs, EUs = stay_data(int(s))
        for k in range(1, len(Bs)):
            a = att_edit(Bs, rs, EUs, k, kappa); isd = any(at['code'] in (ci,cn) and at['kind']==0 for at in Bs[k][1])
            res_['drug' if isd else 'nondrug'].append([a[:4].sum(), a[4], a[5], a[6]])
    for kk, v in res_.items():
        v = np.array(v); print(kappa, kk, 'mean [slots,e,u,z]', v.mean(0).round(5), ' min u', v[:,2].min().round(5), ' max e', v[:,1].max().round(5))
```