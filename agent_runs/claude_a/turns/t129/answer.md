Run 124 settles the norepinephrine numbers in this simulated ICU. After adjusting for the previous dose, MAP and the dosing zone, the response of map_arterial to one extra mcg/kg/min for one hour is about +3.4, +2.6, +1.75, +0.9, +0.4 and +0.2 at lags 1 to 6. That matches my earlier estimate.

The plain factual regression gives a different pattern: +2.4 at lag 1, then negative at later lags. That is the confounding I noted before. Higher earlier doses mark sicker stays with lower MAP. So the record's association differs from the drug's effect, and the original model has learned the association.

My targets stay as they are:
- Insulin x2: about −14.3 on glucose_finger. Insulin x1/2: about +7.2.
- Norepinephrine x2: about +4.1 on map_arterial. Norepinephrine x1/2: about −2.0.
- The original model gives −1.9, +1.0, −0.4 and −0.5.

The next step is to build the edit, starting with insulin. The conversation was restarted, so I first need an exact inventory of what my kernel still holds. That means the replica functions with their signatures, the parameter dictionaries and the target arrays. I also want the start of my notebook, which holds the data facts (dosing policy, tiers, kernels) not shown in the tail above. I expect the validated offline replica functions (rollout_stay, tf_eval, build_ds, head_value, core_step_b) to be present and callable. If they are missing, I will have to rebuild before editing.

```python
import types, inspect
g = {k:v for k,v in list(globals().items()) if not k.startswith('_')}
funcs, others = [], []
for k,v in g.items():
    if isinstance(v, types.ModuleType) or k in ('In','Out','exit','quit','get_ipython'): continue
    if callable(v) and hasattr(v,'__code__'):
        try: sig = str(inspect.signature(v))
        except Exception: sig='?'
        funcs.append(f"{k}{sig}")
    elif not isinstance(v, type):
        t = type(v).__name__
        shp = getattr(v,'shape',None)
        if shp is None and hasattr(v,'__len__'):
            try: shp = len(v)
            except Exception: shp=None
        others.append(f"{k}:{t}{'' if shp is None else tuple(shp) if hasattr(shp,'__iter__') else shp}")
s1 = "; ".join(funcs); s2 = ", ".join(others)
print("FUNCS:", s1[:4500])
print("VARS:", s2[:3500])
notes = open('/work/notes.md').read()
print("NOTES LEN", len(notes))
print(notes[:max(0,len(notes)-8000)][:5000])
```