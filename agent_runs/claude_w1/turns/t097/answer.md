The free-running replica is exact over whole stays: relative error 0.0000 on every bundle, starting from a zero state with a 1-second first gap. It takes about 0.07 s per stay, so I can generate exact model states for thousands of training stays, including stays with counterfactual doses.

That lets me build the dose decoder properly. Regressing the recorded dose on the head input would mostly learn the glucose-policy link (dose ≈ policy(glucose)), which does not respond to a dose edit. So I compute states for about 1,500 training stays in which each infusion's dose is multiplied by an independent random factor between 0.45 and 2.2. This covers the ×2 and ×½ edits. Readings stay as recorded, and a quarter of the stays are left unperturbed.

At every glucose_finger and map_arterial reading I store the head input, the register, and the doses of the infusions started 1, 2 and 3 hours earlier. Then I fit ridge decoders.

How to read the result:
- **Head-level edit is viable:** the latest insulin dose and NE rate are linearly decodable with high R² (say above 0.8) and slope near 1 on held-out stays. A head-level edit, c·(decoded dose − policy-expected dose from the register), is then feasible.
- **Head-level edit is not enough:** R² is low. I would then need an upstream channel through the prelude's action-reading head.

```python
open('/work/notes.md','a').write("""
- Run 84-91: HEAD replica approx (type_emb[1], gap(log dt_s/10), last_numeric=[has,log1p(age_min)/10,zlast,has,has,0,0,0], GELU, last_status row0 = 0): MAE ~2 vs tool (not fully exact; accept).
- FREE-RUN replica EXACT from zero init (h=0,out=0, dt0=1 s): run_stay(B) in kernel, 0.07 s/stay -> can generate exact states locally for any record (incl. counterfactual doses).
- Prelude attention: slot queries attend ~100% to the u token (head2 always; heads0,1 ~97-99% on insulin bundles); z token ~0 attention. => dose path = values of u token, same for all slots.
- PLAN now: head-level edit c*(dose_decoded_from_q - pi_hat(register last value)); decoder trained on states with RANDOMLY PERTURBED doses (breaks policy link).
""")
tr = pd.read_csv('/data/train.csv')
trg = {s: d for s, d in tr.groupby('stay_id')}
train_ids = list(trg.keys())
def wc_vec(prev, dt, code, last):
    h = prev[192:960].reshape(4,192)*np.exp(-dt*np.exp(Wo['core.flow.a']))
    ctx = prev[:192] + LNn(h.mean(0),'core.out_norm') + mlp2n(np.array([np.log(dt)/10]),'gap')
    pr = mlp2n(ctx,'z_prior')
    q = LNn(ctx + linn(pr[:32],'z_head') + Wo['type_emb.weight'][1], 'query_norm')
    has, agemin, zl, vb = last
    return q + Wo['embed.code.weight'][CID[code]] + linn(np.array([has,np.log1p(agemin)/10,zl,has,has,0,0,0],float),'last_numeric') + Wo['embed.bins.value.weight'][vb] + Wo['embed.bins.rate.weight'][0]
def gen_samples(df, fac_rng=None, pert=True):
    df = df.sort_values('hour')
    drug = df[df.event.isin(['insulin','norepinephrine'])]
    fac = {}
    for r in drug.itertuples():
        fac[(r.event, round(r.hour*60))] = float(np.exp(fac_rng.uniform(-0.8,0.8))) if pert else 1.0
    B = bundles_of_stay(df, RU0, dscale=lambda ev, hr: fac[(ev, round(hr*60))])
    St = run_stay(B)
    tk = np.array([b['t'] for b in B])
    ins = {round(r.hour*60): r.rate*fac[('insulin', round(r.hour*60))] for r in drug[drug.event=='insulin'].itertuples()}
    nep = {round(r.hour*60): r.rate*fac[('norepinephrine', round(r.hour*60))] for r in drug[drug.event=='norepinephrine'].itertuples()}
    def lagdose(D, t, L):
        lo, hi = (t-L)*60, (t-L+1)*60
        v = [D[m] for m in D if lo <= m < hi]; return v[-1] if v else 0.0
    out = []; lastv = {}
    for r in df.itertuples():
        if r.event not in MUNIT: continue
        k = int(np.searchsorted(tk, r.hour - 1e-9))
        if r.event in ('glucose_finger','map_arterial') and k>0 and abs(tk[k]-r.hour)<1e-6:
            if r.event in lastv: lv_, lh_ = lastv[r.event]; zl, vb = zb(r.event,'value',lv_); last=(1,(r.hour-lh_)*60,zl,vb)
            else: last=(0,0,0,0); lv_=np.nan
            w = wc_vec(St[k-1], (tk[k]-tk[k-1])*3600, r.event, last)
            D = ins if r.event=='glucose_finger' else nep
            out.append((r.event, r.stay_id, r.hour, lv_, r.value, lagdose(D,r.hour,1), lagdose(D,r.hour,2), lagdose(D,r.hour,3), w))
        lastv[r.event] = (r.value, r.hour)
    return out
t0 = time.time(); rng_f = np.random.default_rng(7); SAMP = []
for i, s in enumerate(train_ids[:1500]):
    SAMP += gen_samples(trg[s], rng_f, pert=(i % 4 != 0))
    if time.time()-t0 > 780: print('time cut at', i); break
print('samples', len(SAMP), 'time %.0f' % (time.time()-t0))
import pickle; pickle.dump(SAMP, open('/work/samp_pert.pkl','wb'))
```