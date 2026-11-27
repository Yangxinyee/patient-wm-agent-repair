The insulin atom embedding is now replicated exactly (max error 0.0 on all 20 coordinates at all 11 rates):

insulin atom = code[8] + kind[0] + ekind[2] + unit[3] + rate_unit[2] + kq[0] + bins.rate[b] + bins.dose[b] + numeric([0, 0, z_dose, 1, z_rate, 1]) + numeric.bias

No action, route, status or discharged_to rows are used.

Next I'll finish the replica for norepinephrine. The NE atom should have the same form with code[9], probably rate_unit[3] and one of the remaining unit rows. Its dose should be 4.8 × rate, which I'll confirm from the records rather than assume. I'll fit it against NE-only bundles, which all have distinct continuous rates. Then I'll check the combined insulin + NE bundles, where the encoder should sum two mlp1 outputs divided by √2 and add size(log 3).

An exact match on both bundle types would mean I can compute u_k for any dose, including the doubled and halved ones the rollout uses. I'll save the confirmed recipe to my notes.

```python
tun_ne = tun_d[tun_d.event=='norepinephrine'].copy()
ne_r = tun_ne.set_index(['stay_id','hour']).rate; ne_r.index = [(a, round(b,4)) for a,b in ne_r.index]
ne_dose = tu[tu.event=='norepinephrine'].set_index(['stay_id','hour']).dose; ne_dose.index = [(a, round(b,4)) for a,b in ne_dose.index]
rn = np.array([ne_r.get(k, np.nan) for k in key4]); dn_ = np.array([ne_dose.get(k, np.nan) for k in key4])
m_no = hn4 & ~hi4
print('NE dose/rate ratio:', np.nanmedian(dn_[m_no]/rn[m_no]), np.nanstd(dn_[m_no]/rn[m_no]).round(4))
idx_no = np.where(m_no & np.isfinite(rn))[0][:60]
TGN = torch.tensor(U[idx_no])
def ne_parts(r, d): return drug_atom_parts('norepinephrine', r, d)
PN = torch.stack([ne_parts(rn[i], dn_[i]) for i in idx_no])
resN = []
for un, ru in itertools.product([0,2,3,4,5],[0,2,3]):
    base = E_['code'][9] + E_['kind'][0] + E_['ekind'][2] + CAT['unit'][un] + CAT['rate_unit'][ru] + E_['kq'][0]
    resN.append((float((u_from_atoms(PN + base)[:, u_c] - TGN).abs().max()), un, ru))
resN.sort(); print('NE best:', resN[:3])
unN, ruN = resN[0][1], resN[0][2]
def ins_atom(r): return drug_atom_parts('insulin', r, r) + E_['code'][8] + E_['kind'][0] + E_['ekind'][2] + CAT['unit'][3] + CAT['rate_unit'][2] + E_['kq'][0]
def ne_atom(r, d=None): return drug_atom_parts('norepinephrine', r, 4.8*r if d is None else d) + E_['code'][9] + E_['kind'][0] + E_['ekind'][2] + CAT['unit'][unN] + CAT['rate_unit'][ruN] + E_['kq'][0]
def u_set(atoms):
    A = torch.stack(atoms); n = len(atoms)
    h1 = gelu(A @ load('encode_action.mlp1.0.weight').T + load('encode_action.mlp1.0.bias')) @ load('encode_action.mlp1.2.weight').T + load('encode_action.mlp1.2.bias')
    s = h1.sum(0)/np.sqrt(n) + load('encode_action.size.weight')[:,0]*np.log1p(n) + load('encode_action.size.bias')
    h2 = gelu(load('encode_action.mlp2.0.weight') @ s + load('encode_action.mlp2.0.bias')) @ load('encode_action.mlp2.2.weight').T + load('encode_action.mlp2.2.bias')
    return LNf(h2, load('encode_action.norm.weight'), load('encode_action.norm.bias'))
m_b = hi4 & hn4; idx_b = np.where(m_b & np.isfinite(rn) & np.isfinite(ri))[0][:80]
errb = max(float((u_set([ins_atom(ri[i]), ne_atom(rn[i], dn_[i])])[u_c] - torch.tensor(U[i])).abs().max()) for i in idx_b)
print('both-bundles max err:', round(errb,5), 'n', len(idx_b))
open('/work/notes.md','a').write("""
## Runs 57-64: PROBE (fwd_4) + exact encode_action replica
- Probe: z_post rewritten so z_mean copies input coords (offset 30); z_in.weight=0. Activation = GELU (act(.5)=.3457).
- z_post input order [ctx, e_k, u_k] confirmed. u_k for no-action bundles = const = LN(mlp2(size.bias)) (not zero).
- u_k depends ONLY on drug rates in bundle (END atoms not in action set). Insulin atom EXACT (err 0):
  code[8]+kind[0]+ekind[2]+unit[3]+rate_unit[2]+kq[0]+bins.rate[b]+bins.dose[b]+numeric([0,0,z_d,1,z_r,1])+bias;
  bin b = 1+#(edges<x); z=(x-median)/iqr clip 10. DeepSets: sum mlp1/sqrt(n) + size(log(1+n)), mlp2, LN eps1e-5.
- u_I(rate) is non-monotonic: rate 1 very different from .5,2,3,4,6 (which are similar) -> weak dose signal.
""")
```