## Step 13: the one-hour insulin effect is stable at about −2 mg/dL per unit/h

**What the last result showed.**
- The insulin coefficient on next-hour finger glucose barely moves as controls are added: −1.99, −2.02, −2.02 (SE about 0.05).
- Once previous readings are controlled for, earlier insulin doses also lower glucose (−1.39 at lag 1). So the effect persists beyond one hour.
- Finger and lab glucose at the same hour agree, with no offset (mean difference 0.03, SD 9.1). They measure the same hidden glucose.
- Arterial MAP runs about 3 mmHg below cuff MAP.

**What I will do next.** Cross-check the −2 estimate with the threshold-discontinuity design, which needs no assumption about the noise.
- Compare next-hour glucose just above and just below the 150, 180, 220 and 260 thresholds of the fresh finger reading, using local linear fits.
- Scale each jump by the matching jump in insulin rate.
- First check that the chance of having a next-hour reading does not itself jump at the thresholds, since that would bias the comparison.

**What would count as evidence.** Discontinuity estimates of about −2 per unit/h would confirm the noise-based estimate. Materially different estimates would mean the within-band noise is confounded.

```python
s = x[(x.age_glucose_finger==0)].copy()
s['is_latest'] = ~(s.age_glucose_lab.fillna(99) < s.age_glucose_finger)
s = s[s.is_latest]
s['has_next'] = s.gf_next.notna()
def rd(s, thr, bw, ycol, dcol='ins_rate'):
    sub = s[(s.glucose_finger-thr).abs().le(bw) & (s.glucose_finger!=thr)]
    out = {}
    for col in [ycol, dcol, 'has_next']:
        ss = sub[sub[col].notna()]
        r = ss.glucose_finger.values - thr
        above = (r>0).astype(float)
        res = ols(ss[col].values.astype(float), [above, r, r*above], ['jump','r','r_above'])
        out[col] = (res.loc['jump','b'], res.loc['jump','se'], len(ss))
    return out
for thr in [150,180,220,260]:
    for bw in [8,15]:
        o = rd(s, thr, bw, 'gf_next')
        jy, sey, n = o['gf_next']; jd, sed, _ = o['ins_rate']
        print(thr, bw, "jump next-glucose %.2f (%.2f) n=%d | jump rate %.3f (%.3f) | ratio %.2f | jump has_next %.3f (%.3f)" % (jy, sey, n, jd, sed, jy/jd, o['has_next'][0], o['has_next'][1]))
```