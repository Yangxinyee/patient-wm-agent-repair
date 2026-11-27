My explicit step function reproduces the stored states (max error 0.011 on an h scale of about 35; `out` error 1.4e-4).

The attention measurement found a clean channel:

| Bundle type | Prelude head 3, slot → u attention |
|---|---|
| insulin start (90 bundles) | 1.00, sd 0.00 |
| norepinephrine start (659) | 1.00, sd 0.00 |
| both drugs (1735) | 1.00, sd 0.00 |
| measurement-only | ≈0 (attention goes to e, about 0.83) |

Other observations:
- The z_in token gets essentially no attention.
- At action bundles, no head attends to the slots. So in the prelude, state information reaches the slot outputs only through the residual stream and the FFN.

**Conclusion.** Head 3 gives every slot exactly the value vector of the u token at action bundles. A rate signal can therefore be written with a fixed gain, if the head-3 value space carries the rate roughly linearly.

**Plan for the memory.** Repurpose a few h dimensions:
- memory dimensions with their own decay;
- a pair of "carrier" dimensions written with opposite signs, so the constant parts cancel exactly through `prelude_norm` and `b_in`.

This only works if I can find h dimensions whose original content matters little.

**This run.**
1. Rank h dimensions by how much the model reads them: the LayerNorm gains of every reader of h, plus their spread and timescale.
2. Test how well the insulin and norepinephrine rates can be decoded linearly from the head-3 value of the u token.
3. Check how much the LayerNorm gain (1/σ) varies on the write and read paths.

**What would support the plan:** dimensions that every reader barely uses, rate decodable with R² near 0.99, and LayerNorm σ varying by about 10% or less. **What would not:** poor decodability or highly variable σ. Then I would need to engineer a dedicated linear rate path in the action encoder.

```python
P_ = PO
# (a)-(d) dimension importance stats
hs = HS.reshape(-1, 192)
std_h = hs.std(0); mean_h = hs.mean(0)
gam = {n: P_[n].abs() for n in ['core.prelude.0.norm1.weight','core.prelude_norm.weight','core.blocks.0.0.norm1.weight','core.blocks.0.0.norm2.weight','core.blocks.0.0.norm3.weight','core.blocks.0.1.norm1.weight','core.blocks.0.1.norm2.weight','core.blocks.0.1.norm3.weight','core.coda.0.norm1.weight','core.coda.0.norm3.weight','core.out_norm.weight','query_norm.weight']}
G_ = torch.stack(list(gam.values()))
tau_h = 1.0/torch.exp(P_['core.flow.a'])/3600.0  # hours
sp = F.softplus(P_['core.delta_in']); abar = torch.exp(-sp*torch.exp(P_['core.a_in']))
# 'visibility' of dim j: its std relative to the slot-token sd times the LN gains of readers
sig_tok = hs.std(1).mean()
vis = std_h / sig_tok * G_[[0,2,5,8,10]].mean(0)
order = torch.argsort(vis)
print('token sd', round(sig_tok.item(),2), ' std_h quantiles', np.round(np.quantile(std_h.numpy(), [0,.1,.5,.9,1]),2))
print('gamma quantiles per LN (min, 10%, median):', {k.replace('core.','').replace('.weight',''): tuple(np.round(np.quantile(v.numpy(), [0,.1,.5]),3)) for k,v in gam.items()})
print('lowest-visibility dims:')
for j in order[:14].tolist():
    print(j, f"std {std_h[j]:.2f} mean {mean_h[j]:.1f} tau_h {tau_h[j]:.2f} abar {abar[j]:.3f} sp {sp[j]:.3f} gam", np.round(G_[:, j].numpy(), 2).tolist())
# LN sigma variability
print('CV of sd(mean_slots h) over bundles', (HS.mean(1).std(1).std()/HS.mean(1).std(1).mean()).item())
print('CV of sd(slot token h) over bundles', (HS.std(2).std()/HS.std(2).mean()).item())
# (e) decodability of rate from head-3 value of the u token
def rates_by_bundle(at_):
    a0_ = at_[at_['kind'] == 0]
    ri = a0_[a0_['event'] == 'insulin'].groupby(['stay_id','minute'])['rate'].sum()
    rn = a0_[a0_['event'] == 'norepinephrine'].groupby(['stay_id','minute'])['rate'].sum()
    idx = pd.MultiIndex.from_arrays([SB['stay_id'].values, SB['minute'].values])
    return np.nan_to_num(ri.reindex(idx).values), np.nan_to_num(rn.reindex(idx).values)
r_ins, r_ne = rates_by_bundle(at2)
act = (r_ins > 0) | (r_ne > 0)
n1u = LN(uALL, 'core.prelude.0.norm1', P_)
Wv = P_['core.prelude.0.attn.kv.weight'][192:]
V3 = n1u @ Wv[144:192].T
from sklearn.linear_model import LinearRegression
for nm_, r_ in [('insulin', r_ins), ('norepi', r_ne)]:
    X_ = V3[act].numpy(); y_ = r_[act]
    lr = LinearRegression().fit(X_, y_)
    print(nm_, 'R2 from head-3 value (48d):', round(lr.score(X_, y_), 4), ' rate range', y_.min(), y_.max())
    X2_ = n1u[act].numpy(); lr2 = LinearRegression().fit(X2_, y_)
    print(nm_, 'R2 from full LN1(u) (192d):', round(lr2.score(X2_, y_), 4))
```