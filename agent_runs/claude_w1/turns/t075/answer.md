The z posterior replica is now essentially exact for drug bundles, at 0.8% relative error. The working configuration:
- dose unit "mg" uses row 4 and "units" row 5;
- rate unit "mcg/kg/min" uses row 3 and "units/hour" row 2;
- infusion END atoms go through the measurement/event encoder, not the action encoder;
- empty sets are encoded as LN(mlp2(size(0))).

I'll record this in the notebook now, since it's the foundation for everything after. Then I'll check all bundle types with the final configuration to see whether reading-only bundles still have the 0.135 residual.

```python
UM={'mg/dL':3,'mmHg':2,'mg':4,'units':5}; RM={'mcg/kg/min':3,'units/hour':2}
with open('/work/notes.md','a') as fh:
    fh.write("""
## Runs 42-69: replicas
- Dose innovation decodability (tuning states): insulin nu1 R2 ~0.24 from head query q (slope 0.29); NE innovation R2 0.09 (slope 0.10).
  => pure readout edit too noisy (esp. NE). Need upstream signal.
- Original model rollout insulin x2 per hour: -0.53 (h13) .. -2.28 (h24); truth -3.2 .. -18.8. Model persistence ~0.78/h.
- HEAD replica (from states): view=out+LN_out(mean_slots(h*exp(-dt*exp(flow.a))))+gap(log(dt_s)/10)+z_head(z_prior mean); q=LN_query(view+type_emb[1]);
  with_code=q+code+last_numeric([has,log1p(age_min)/10,zlast,has,has,0,0,0])+bins.value[lastbin]; GELU; MAE 2.4 vs tool (noise floor ~1.2).
- ENCODER/z_post replica EXACT (rel err 0.008 on drug bundles): GELU everywhere; empty set -> LN(mlp2(size(0))) (not zero);
  units: mg/dL->3, mmHg->2, mg->4, units->5; rate_unit: mcg/kg/min->3, units/hour->2; drug END atoms go to e (measurement encoder), not u.
  atom = code+kind+ekind+kq[0]+unit+rate_unit+bins+numeric@[zv,hv,zd,hd,zr,hr]+bias. z_post([ctx,e,u]) ctx=out+LN_out(mean h decayed)+gap(log dt/10).
""")
for t_ in ['M','A','']:
    print(t_ or 'none', zpost_batch4(UM,RM,types={t_},end_attr=False,end_in_action=False,nst=20))
```