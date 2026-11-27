`step_b` returns (new h, `out`, intermediates). The intermediates include E (stays × 4 slots × 192), the time-flowed h, the prelude's 7×7 attention weights over [4 slots, e, u, z] and `hn`. One batched call is made per bundle position, and `srows` maps positions to bundle rows. So I can collect E for every bundle.

**Target for the insulin memory.** The insulin policy depends only on the current glucose tier, with an independent random multiplier. So the memory should accumulate:
- "rate − tier mean" at insulin bundles: the tier mean is written as −rbar(tier) at the glucose-reading bundle, and the rate as +rate at the infusion bundle five minutes later.

That is zero-mean on factual records, so it cannot bias the factual predictions. Under ×2 it still grows by the full recorded rate.

**Norepinephrine needs a different baseline.** I worked out that a "deviation from expected titration given the previous NE dose" would cancel out under ×2. The expected dose scales with the doubled previous dose, so it would erase the counterfactual. I will handle NE after insulin works.

**Next step:** build a 300-stay training dataset (disjoint from sel2) and capture E per bundle. I will check that the captured states line up with `tf_states_ds`'s output, and look at the atom table columns so I can build per-bundle dose and glucose targets.

**What success looks like:** exact alignment (max diff ≈ 0) and identifiable insulin and glucose atoms.

```python
import time
rng_ = np.random.default_rng(7)
all_tr = np.array(sorted(tr['stay_id'].unique()))
poolA = np.setdiff1d(all_tr, np.array(sel2))
stA = list(rng_.choice(poolA, 300, replace=False))
t0=time.time(); dsA = build_ds(tr, stA, 0)
CAP=[]
def step_b(hprev, e, u, z, dt, Mx, Mage, Mmask, P, ret=False):
    r1 = _orig_step_b(hprev, e, u, z, dt, Mx, Mage, Mmask, P, ret=True); CAP.append(r1)
    return (r1[0], r1[1]) if not ret else r1
try:
    HnA, OnA = tf_states_ds(P, dsA)
finally:
    step_b = _orig_step_b
nbA = dsA['e'].shape[0]
EA = torch.zeros(nbA, 4, 192); YA = torch.zeros(nbA, 4, 192)
for s_i, rows_s in enumerate(dsA['srows']):
    for k, rw in enumerate(rows_s):
        EA[rw] = CAP[k][2]['E'][s_i]; YA[rw] = CAP[k][0][s_i]
print("built %.1fs"%(time.time()-t0), "bundles", nbA, " align max|dY|", float((YA-HnA).abs().max()))
print(dsA['at'].columns.tolist()); print(dsA['at'].head(8).to_string())
print(dsA['bd'].columns.tolist())
```