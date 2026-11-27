mod_events takes a dict from drug name to factor and rescales the rate and dose fields of those drug events. Rows are namedtuples with stay_id, hour, event, value, rate, dose, and so on. So for the exact counterfactual record I can rebind the global mod_events, which run_cf looks up by name, to a version that also adds a per-(code, hour) shift to reading values. The head's last-value features come from the reading table rt, so those must also be shifted for counterfactual rows.

Next I check:
- the columns of RT and what last_feats uses;
- the output shapes of run_cf and heads_on_states on one stay, with timing.

If run_cf returns per-bundle (O, H) quickly, under about 0.5 s per stay, I can afford about 15 counterfactual variants per drug on the 100 tuning stays, plus more training stays.

```python
import time
print(list(RT.columns)); peek('last_feats'); peek('X_of'); peek('solve_theta')
s=2020000000; t0=time.time(); r=run_cf(s,{'insulin':2.0},12*60,Xs); t1=time.time()
print(type(r).__name__, [tuple(x.shape) if hasattr(x,'shape') else type(x).__name__ for x in (r if isinstance(r,(tuple,list)) else [r])], 'sec %.2f'%(t1-t0))
rt=RT[RT.stay_id==s]; print(len(rt), rt[['b','dt']].head(2).values.tolist() if 'b' in rt else '')
```