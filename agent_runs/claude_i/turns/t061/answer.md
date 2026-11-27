The old functions show what last session was attempting. `zpost_cand` used code 11 (hospital_admission) and the out_norm bias. So it was matching the very first bundle, whose prior state (initial h and out) is unknown. That likely explains the imperfect 0.976 match, rather than a wrong encoder.

`atom_vec` sums code, kind, event-kind and kq rows, the unit and rate_unit tables, the three bin tables and the numeric map. I'll rebuild the encoders cleanly and validate them on later bundles, where the previous state is stored and ctx is already validated through the head replica. For each later bundle, z_post([ctx, e_k, u_k]) must reproduce the stored posterior mean.

Next, I will look at which embedding-table rows are exactly zero, i.e. padding rows. That shows which categorical lookups contribute nothing and shrinks the guesswork about index mappings. Zero row-0 entries mean index 0 is "absent".

```python
for nm in ['embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight','embed.code.weight',
           'embed.categorical.action.weight','embed.categorical.discharged_to.weight','embed.categorical.rate_unit.weight',
           'embed.categorical.route.weight','embed.categorical.status.weight','embed.categorical.unit.weight',
           'embed.bins.value.weight','embed.bins.rate.weight','embed.bins.dose.weight','last_status.weight','type_emb.weight']:
    w_ = Tn(nm); print(nm, np.round(np.linalg.norm(w_,axis=1),2))
```