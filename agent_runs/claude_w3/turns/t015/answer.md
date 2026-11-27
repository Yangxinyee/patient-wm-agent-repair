The instrumented impulse response gives a clear, consistent shape. A unit/h of insulin lowers glucose by about 2.1 after one hour and by about 2.4–2.8 after two hours. The lag-1 value is smaller because the infusion is still running when the next reading is taken. The effect stays near −2 to −2.5 through lag 4, then fades: about −1 at lags 5–6, −0.5 to −1 at lags 7–10, and about 0 by lag 12. So insulin's effect is transient, lasting roughly 4 hours at full strength and then washing out. It does not shift the near-random-walk glucose level permanently. Long-lag estimates are noisy (se 0.2–0.3) and vary somewhat between windows, so I will refine them later.

Next I will record this in the notebook and turn the response into a provisional "true effect" for the readouts: for each hour t from 13 to 24, effect = Σ θ(t−s) × extra dose at s. I will compute this on the tuning stays and on train, both pooled over stay-hours and as a mean of stay means. I will also pick 100 tuning stays that have drug from hour 12 on, for a forward pass of the original model. Comparing the model's answers with this provisional truth will show how wrong it currently is and in which direction.

```python
notes_add = """
## Runs 2-14: data structure and insulin policy (train.csv)
- 30k stays, hours -2..25.17, readings on the hour, infusions start at h+0.0833 and run 1h. All stays same length.
- Insulin POLICY (memoryless): at an hour with a fresh glucose (finger preferred, else lab), if >150 -> rate = base*jitter,
  base = 1 (150-180], 2 (180-220], 3 (220-260], 4 (260-300], 6 (>300). Jitter multiplicative, mean 1.056, sd ~0.33, rounded to 0.5.
  Jitter: lag-1 autocorr 0.0007, no stay component, uncorrelated with trend and with simultaneous lab|finger -> as-good-as-random.
- Measurement: finger hourly when last policy glucose >150; else finger w.p. ~0.30 at random. Lab random 20% of hours.
  lab & finger measure same latent glucose; noise roughly proportional to level; finger noisier.
- Never-insulin stays: glucose ~ random-walk-like (variogram rising ~8/h over 15h).
- IV (jitter instruments, lab outcome, later doses as endogenous): insulin impulse response per unit/h:
  lag1 ~ -2.1, lag2-4 ~ -2.4 (-2.0..-2.8), lag5-6 ~ -1, lag7-10 ~ -0.5..-1, lag12 ~ 0. Transient effect, NOT permanent.
"""
open('/work/notes.md','a').write(notes_add)

def build_panel(df):
    m = df[df.event.isin(meas_codes)].copy(); m['h'] = m.hour.round().astype(int)
    P = m.pivot_table(index=['stay_id','h'], columns='event', values='value', aggfunc='first')
    d = df[df.event.isin(['insulin','norepinephrine'])].copy(); d['h'] = np.floor(d.hour).astype(int)
    D = d.pivot_table(index=['stay_id','h'], columns='event', values='rate', aggfunc='first')
    stays = df.stay_id.unique()
    idx = pd.MultiIndex.from_product([stays, np.arange(0,25)], names=['stay_id','h'])
    p = pd.DataFrame(index=idx).join(P).join(D)
    for c in meas_codes+['insulin','norepinephrine']:
        if c not in p: p[c]=np.nan
    p[['insulin','norepinephrine']] = p[['insulin','norepinephrine']].fillna(0.0)
    return p.reset_index().sort_values(['stay_id','h'])
tup = build_panel(tu)
theta = {1:-2.2, 2:-2.4, 3:-2.4, 4:-2.3, 5:-1.5, 6:-1.1, 7:-0.9, 8:-0.8, 9:-0.6, 10:-0.5, 11:-0.3, 12:-0.1}
def prelim_effect(p, drug='insulin', th=theta, mult=2.0):
    S = p.stay_id.nunique(); D = p[drug].values.reshape(S, 25); ids = p.stay_id.values.reshape(S,25)[:,0]
    eff = np.full((S, 25), np.nan)
    for t in range(13, 25):
        e = np.zeros(S); elig = np.zeros(S, bool)
        for s in range(12, t):
            e += th.get(t-s, 0.0) * (mult-1) * D[:, s]
            elig |= D[:, s] > 0
        eff[:, t] = np.where(elig, e, np.nan)
    pooled = np.nanmean(eff[:,13:]); 
    with np.errstate(all='ignore'):
        sm = np.nanmean(eff[:,13:], axis=1)
    return pooled, np.nanmean(sm), np.isfinite(sm).sum(), ids, eff
for nm, p in [('tuning', tup), ('train', pn0)]:
    for mult in [2.0, 0.5]:
        pooled, stm, ns, _, _ = prelim_effect(p, mult=mult)
        print(nm, "insulin x", mult, ": pooled stay-hour mean", round(pooled,2), " mean of stay means", round(stm,2), " n stays", ns)
# choose 100 tuning stays with insulin or norepi at/after hour 12
S_t = tup.stay_id.nunique(); Dins = tup.insulin.values.reshape(S_t,25); Dne = tup.norepinephrine.values.reshape(S_t,25)
ids_t = tup.stay_id.values.reshape(S_t,25)[:,0]
has_ins = (Dins[:,12:]>0).any(1); has_ne = (Dne[:,12:]>0).any(1)
print("tuning stays with insulin>=12:", has_ins.sum(), " norepi>=12:", has_ne.sum(), " both:", (has_ins&has_ne).sum(), " either:", (has_ins|has_ne).sum())
```