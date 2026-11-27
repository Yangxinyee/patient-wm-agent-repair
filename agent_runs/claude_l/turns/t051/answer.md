The vocab and bundle structures are as I remembered. Each bundle is a list of atoms with code, kind, ekind, unit, rate_unit, the value/dose/rate fine bins and six numeric features `(z_value, has_value, z_dose, has_dose, z_rate, has_rate)`. These match the states' bundle times and atom counts, for example 1, 3, 2 and 1 atoms at hours −2, −1, 0 and 0.083.

The bin tables explain part of the norepinephrine anomaly. NE rates are capped at 1.0 in the records, so any NE x2 rate above 1.0 lands in bin 16. That bin was never seen for NE, and the dose and rate bin tables are shared with insulin. Insulin x2 above 5 U/h also saturates at bin 16. Only the numeric z-scores extrapolate linearly.

I also redid the factual-fit arithmetic. The glucose noise is large (residual sd about 12.5 mg/dL), so a modest systematic bias of about 1 mg/dL costs only about 0.003 nats per glucose reading. That leaves some room for simpler gain-type edits, but these cannot create the multi-hour memory or the recovery dynamics. An exact reimplementation would still be far better.

Before deciding how much to invest, I want to see how close my last reimplementation attempt came: what quantity it matched, with which error metric, and which options were best. Its variables are still in the kernel.

```python
for nm_ in ['best','results','errs','cands','zcands','samples','E_list','G_list','ae','az','div','unit_name','um','rum','V_unit','CODE_ID','NUM','ACTS','ACTS2','order','parts','gname','gf','zt','nm','na','eps','r2','recon','u_','hs']:
    if nm_ in globals():
        o = globals()[nm_]
        try:
            if isinstance(o, np.ndarray): print(nm_, 'ND', o.shape, np.round(o.ravel()[:6], 4))
            elif isinstance(o, torch.Tensor): print(nm_, 'T', tuple(o.shape), o.ravel()[:6])
            elif isinstance(o, dict): print(nm_, 'dict', len(o), str(list(o.items())[:4])[:400])
            elif isinstance(o, (list, tuple)): print(nm_, type(o).__name__, len(o), str(o[:3])[:400])
            elif callable(o): print(nm_, 'callable')
            else: print(nm_, type(o).__name__, str(o)[:200])
        except Exception as ex: print(nm_, 'ERR', ex)
```