The kernel holds the factual building blocks:
- the reading table RT for the 100 tuning stays, with head queries Q;
- the replica states Xs/Xt;
- BUN, the bundles as (minute, events) lists, where each event is an (event, kind, row) tuple;
- truth tables (R, de).

I see no stored counterfactual-state arrays or fitted unit weights, so whatever amp/unit fits I did earlier are not reusable as data. The quickest route is to reuse run_cf, mod_events and heads_on_states. Before that I need to know what form `edits` takes, and whether it can also shift reading values, which the exact counterfactual record needs.

Next I inspect the referenced names and constants of mod_events, run_cf and true_shift.
- If mod_events supports multiplying drug rates and doses, I will add a reading-shift step myself.
- If it already shifts readings, I reuse it as is.

```python
def peek(nm):
    c=globals()[nm].__code__
    cs=[x for x in c.co_consts if isinstance(x,(str,int,float)) and not (isinstance(x,str) and len(x)>40)]
    print(nm,'| names',c.co_names[:30],'| consts',cs[:25],'| vars',c.co_varnames[:20])
for nm in ['mod_events','run_cf','true_shift','meas_atom','bvd']: peek(nm)
s0=2020000000; ev=[e for (m,evs) in BUN[s0] for e in evs]
print(type(ev[0][2]).__name__, [f for f in getattr(ev[0][2],'_fields',getattr(ev[0][2],'index',[]))][:12], sorted(set((e[0],e[1]) for e in ev)))
```