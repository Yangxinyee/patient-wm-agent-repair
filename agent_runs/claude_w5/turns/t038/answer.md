The recurrence is not a simple leaky integrator, so a dose signal written into it would not decay in any easily predictable way:
- `c_out` is a general matrix: diagonal mean −0.016, distance from identity about 1.24.
- The per-step retention a_bar is only 0.28–0.45, so h is constantly re-driven through the normalise → `b_in` → coda loop.
- The h slots carry large, fairly constant components (means up to ±215 per dimension, sd across bundles about 9).

Information from u_k enters as small perturbations on top of these. That may be why the dose values barely register.

Before committing to a costly approach such as re-implementing the whole core, I'll check whether the dose pathway is simply weak or deliberately cut at the input. I'll inspect:
- the `embed.numeric` columns for z_dose and z_rate, compared with z_value;
- the `embed.bins.rate` and `embed.bins.dose` tables;
- the `encode_action` first-layer response to those directions.

Near-zero dose columns or bin tables would mean dose information never enters, and the fix starts at the embedding.

```python
Wn = L('embed.numeric.weight'); bn = L('embed.numeric.bias')
print("numeric col norms [z_value, has_value, z_dose, has_dose, z_rate, has_rate]:", np.round(np.linalg.norm(Wn,axis=0),3))
for nm in ['embed.bins.value.weight','embed.bins.dose.weight','embed.bins.rate.weight']:
    B = L(nm); print(nm, "row norms:", np.round(np.linalg.norm(B,axis=1),2))
    print("   successive-row diff norms:", np.round(np.linalg.norm(np.diff(B[1:],axis=0),axis=1),2))
Ec = L('embed.code.weight'); print("code emb norms", np.round(np.linalg.norm(Ec,axis=1),2))
A1 = L('encode_action.mlp1.0.weight'); M1 = L('encode.mlp1.0.weight')
for j, nmj in enumerate(['z_value','has_value','z_dose','has_dose','z_rate','has_rate']):
    print(nmj, " |A1 @ col| =", round(np.linalg.norm(A1@Wn[:,j]),3), " |M1 @ col| =", round(np.linalg.norm(M1@Wn[:,j]),3))
for nm in ['embed.categorical.unit.weight','embed.categorical.rate_unit.weight','embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight','embed.categorical.action.weight','embed.categorical.route.weight','embed.categorical.status.weight']:
    B = L(nm); print(nm, np.round(np.linalg.norm(B,axis=1),3))
```