## Step 20: the insulin effect is linear and the same at every glucose level

**What the last result showed.** Inside each band, the one-hour effect per extra unit/h is about −1.7 to −2.3 at every rate level (0.5 up to 5 units/h). It is similar in bands 1 and 2; band 3 is noisier but consistent. So the insulin effect is linear in dose and does not depend on the glucose level.

**What I now believe about insulin.**
- The effect is a linear, time-invariant kernel: −1.6 to −1.9 at lag 1, about −2.2 at lag 2, then roughly 25% decay per hour.
- A sustained extra 1 unit/h accumulates to about −11 mg/dL by 12 hours.
- Finger and lab glucose read the same hidden glucose, so the effect should be the same on both.

**What I will do next.** Turn to norepinephrine. First work out its dosing policy:
- what triggers starting (which MAP reading, which threshold);
- how the rate is titrated hour to hour (the change in rate against the latest MAP);
- stopping rules;
- whether there is per-record noise I can use, as with insulin.

**What would count as evidence.** A rule such as "start if MAP < 65; Δrate = f(MAP) + noise; cap at 1.0" with i.i.d. noise would let me reuse the surprise-instrument design. If NE dosing has no noise, I will need the thresholds (a discontinuity design) or the cap.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 15-20: insulin dynamics
- glucose_lab measured randomly p=0.2/hour, independent of state -> unbiased outcome. finger hourly iff prev latest G>150 else p=0.3.
- 2SLS impulse response (dose surprises e=rate-E[rate|band] as instruments), LAB outcome, K=12:
  h=[-1.63,-1.83,-1.78,-1.4,-1.41,-0.89,-0.54,-0.71,-0.12,-0.33,-0.31,-0.03] (se~.3); cumsum ~ -11 at 12h.
  FINGER K=8: [-1.89,-2.42,-2.08,-1.66,-1.25,-0.9,-0.68,-0.48] (se .1).
- One-hour effect linear in rate (per-unit ~ -1.7..-2.3 for rates 0.5..5) and similar across glucose bands -> linear, homogeneous.
""")
xn = xs.copy()
for c in ['map_arterial','map_cuff']:
    xn[c+'_l1'] = xn.groupby('stay_id')['last_'+c].shift(0)
# latest MAP: prefer arterial if same hour?
af = xn.age_map_arterial.fillna(99); ac = xn.age_map_cuff.fillna(99)
xn['lm_art_pref'] = np.where(af<=ac, xn.last_map_arterial, xn.last_map_cuff)
xn['lm_cuff_pref'] = np.where(ac<=af, xn.last_map_cuff, xn.last_map_arterial)
xn['ne_prev'] = xn.prev_ne_rate
xn['dne'] = xn.ne_rate - xn.ne_prev
# starts: previously off, now?
st = xn[(xn.ne_prev==0)]
for col in ['lm_art_pref','lm_cuff_pref','last_map_arterial','last_map_cuff']:
    sub = st[st[col].notna()]
    tb = sub.groupby(pd.cut(sub[col],[0,55,60,62,63,64,65,66,67,68,70,75,200])).ne_rate.apply(lambda s:(s>0).mean())
    print("START prob by", col, tb.round(3).to_dict())
```