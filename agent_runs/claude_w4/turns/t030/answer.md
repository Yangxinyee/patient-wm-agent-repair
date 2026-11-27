## Step 29: true effects for both drugs are now estimated

**What the last result showed.**

*Insulin, with the clean instruments.*
- Lab outcome: −1.5, −1.8, −1.7, −1.3, −1.3, −0.8, −0.45, −0.6, then about 0. The cumulative effect is −10.2 mg/dL per sustained unit/h by 12 hours.
- Finger outcome: −1.8, −2.1, −1.9, −1.55, −1.1, −0.76, −0.44, −0.28, … with cumulative −10.3. This agrees with the lab outcome.

*Norepinephrine on MAP, per mcg/kg/min.* (The print labels in the last output were wrong; the order of the three runs was arterial in arterial-line stays, cuff in non-arterial stays, cuff in arterial-line stays.)

| reading and stays | lag 1 | lag 2 | lag 3 | lag 4 | cumulative |
|---|---|---|---|---|---|
| arterial MAP, arterial-line stays | +3.7 | +2.7 | +1.3 | +1.1 | about +10 |
| cuff MAP, non-arterial stays | +2.7 | +2.0 | +2.0 | +0.85 | about +7 |
| cuff MAP, arterial-line stays | | | | | about +10 (noisier) |

So norepinephrine does raise MAP, and the effect decays faster than insulin's. At typical rates (about 0.5), doubling adds only about 5 mmHg at steady state. The non-arterial population may respond less.

**What I will do next.** Before refining further (norepinephrine linearity across rates, heterogeneity), I will choose tuning stays for a first forward pass. That pass shows what the original model predicts, and in which output format and at which stay-hours. I need stays with both drugs running from hour 12, and arterial lines for the arterial-MAP readout.

**What would count as evidence.** Not a hypothesis test yet; this step is choosing a sample that covers both drugs.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 21-29: NE policy & effects
- NE policy: start iff latest MAP (arterial preferred same hour) < 65 (h>0; hour-0 'starts' are arrivals on NE).
  On: log-ratio steps, drift depends on band (<65 up ~+0.05 additive, 65-75 hold, >75 down ~-0.04 & stop p~.19) AND smoothly on MAP value; iid noise (log-sd ~.3).
- MAP measured state-independently: art-line stays (14.5k, from hour 0) art p=.95, cuff p=.3; other stays cuff p=.9.
- Crossfit HGB expected-dose models (feats prev rate, latest reading, band, age, source, h) -> clean surprises e_ins2, e_ne2 (no within-band corr).
- INS->LAB K12: [-1.53,-1.79,-1.69,-1.32,-1.28,-.81,-.45,-.58,-.09,-.30,-.28,-.08] cum -10.2; INS->FINGER: [-1.83,-2.12,-1.90,-1.55,-1.08,-.76,-.44,-.28,-.21,-.08,-.12,.06] cum -10.3
- NE->MAP_ART (art stays) per mcg/kg/min: [3.71,2.71,1.34,1.06,.43,.79,.17,-.18,...] cum ~10; NE->CUFF non-art stays cum ~7; cuff in art stays cum ~10 (noisy).
""")
gT = add_last(PT)
tsid = PT.reset_index().stay_id.unique()
gt = gT.copy()
info = gt.groupby('stay_id').apply(lambda d: pd.Series({
    'ins_12on': (d[d.h>=12].ins_rate>0).sum(), 'ne_12on': (d[d.h>=12].ne_rate>0).sum(),
    'art': d.map_arterial.notna().any(), 'ins_any': (d.ins_rate>0).sum(), 'ne_any': (d.ne_rate>0).sum()}))
print(info.describe().round(2).to_string())
print("stays with ins from 12 on:", (info.ins_12on>0).sum(), " NE from 12 on:", (info.ne_12on>0).sum(), " both:", ((info.ins_12on>0)&(info.ne_12on>0)).sum(), " NE&art:", ((info.ne_12on>0)&info.art).sum())
```