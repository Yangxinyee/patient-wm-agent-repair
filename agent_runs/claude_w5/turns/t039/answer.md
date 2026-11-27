The dose pathway is not cut at the input:
- The z_dose and z_rate columns of `embed.numeric` have norms of about 3.
- The action encoder's first layer responds strongly to them (norms of about 8 after projection).
- The rate and dose bin tables are fully trained.

So u_k does encode the dose, and the core is what fails to carry it into glucose and MAP predictions.

Several categorical rows have norms of about 0.27–0.28, like untrained initial values. Those are the unused entries:
- index 1 of the unit, rate_unit, text, action, route and status tables;
- kq rows 1–3;
- kind 1 (REVEAL).

That narrows the embedding mapping a lot.

My analysis also shows why a simple edit is risky. The original model has absorbed the average dose effect into its dependence on readings. Any injected dose effect must respond to dose *deviations*, or factual NLL breaks: I estimate about 0.008–0.03 nats for crude versions, mainly from norepinephrine. To design and test such an edit safely I need a local copy of the model.

So I'll re-implement the forward pass in the sandbox and check it against the states file. The first step is to rebuild the exact bundle and atom sequence the model sees, including the trailing END and stay-end bundles. I'll compare per-bundle atom counts and drug flags with the states file for all 100 stays. A full match confirms the tokenisation; a mismatch shows what I've got wrong.

```python
def make_bundles(df_stay):
    atoms = []
    for _, r in df_stay.iterrows():
        t = round(r.hour*60)
        ev = r.event
        if ev in ('glucose_finger','glucose_lab','map_arterial','map_cuff'):
            atoms.append((t, dict(code=ev, kind=0, value=r.value)))
        elif ev in ('insulin','norepinephrine'):
            atoms.append((t, dict(code=ev, kind=0, rate=r.rate, dose=r.dose)))
            atoms.append((round(r.end_hour*60), dict(code=ev, kind=2, rate=r.rate, dose=r.dose)))
        else:
            atoms.append((t, dict(code=ev, kind=0)))
            atoms.append((round(r.end_hour*60), dict(code=ev, kind=2)))
    bund = {}
    for t, a in atoms: bund.setdefault(t, []).append(a)
    return [(t/60, bund[t]) for t in sorted(bund)]
s0 = st_sid[0]
B0 = make_bundles(tr[tr.stay_id==s0])
print(len(B0), (st_sid==s0).sum())
print([ (round(t,4), len(a)) for t,a in B0][-6:])
print(list(zip(np.round(st_hr[st_sid==s0][-6:],4), st_na[st_sid==s0][-6:], st_ins[st_sid==s0][-6:], st_ne[st_sid==s0][-6:])))
print(B0[-1], B0[-2])
```