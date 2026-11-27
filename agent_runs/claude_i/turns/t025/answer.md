The norepinephrine direct kernel is fast and short-lived. Per 1 mcg/kg/min-hour it is about +3.2 at 1 h, +2.6 at 2 h, +1.6 at 3 h, +0.8 at 4 h, +0.3–1.0 at 5 h, then about 0. The sum is about +8.6 mmHg·h.

Rates typically sit around 0.5, so doubling should give roughly +4 mmHg at steady state. The original model says only +0.37 for ×2 and −0.06 for ×½. For insulin, the kernel sum of about −12 per u/h-hour suggests the true ×2 effect is several times the model's −2.0.

The scored edits ×2 and ×3 push norepinephrine past its 1.0 cap, so the dose-response shape matters. I'll check whether the norepinephrine and insulin effects are linear or saturating by estimating the lag-1/lag-2 slope within bins of the previous rate. I'll also save these findings to the notebook first.

```python
notes = """
## Findings so far (session 1)
- Stays: all 25.17h long; readings on the hour (-1..24), drug records at h+5min lasting 1h. Hosp adm always at -2h. NE dose = 4.8*rate (80kg fixed); insulin dose = rate.
- INSULIN POLICY: decision reading = gf at hour h if present else gl at h (else no insulin). Given iff reading >=150 (unrounded).
  rate = round_0.5(base * LogNormal(0, 0.30)), base 1 (150-180), 2 (180-220), 3 (220-250), higher above. Noise iid (autocorr 0.005).
- MEASUREMENT: gf hourly (96%) after insulin, else iid 30%. gl iid 19-20% (MCAR, use for selection-free outcomes). gf-gl sd 9, no bias.
  ~52% of stays have no arterial line; art-line stays have ma ~every hour.
- NE POLICY: start if MAP<~62-65 (ma preferred, else mc); continuing: rate_t = rate_{t-1}*exp(step + 0.30*xi), step ~ +0.1..0.15 (<65), 0 (65-75), -0.22 (>75); cap 1.0; stop sometimes if MAP>75.
- INSULIN EFFECT (direct kernel per 1 u/h for 1h dose, feedback-deconvolved, gl outcome): K1..K12 ~ -2.0,-2.35,-2.1,-1.6,-1.3,-0.9,-0.6,-0.5,-0.4,-0.1,0,0. Sum ~ -12. Linear in rate (rounding of lognormal explains low-end compression).
- NE EFFECT (direct kernel per 1 mcg/kg/min-hour, ma outcome): K1..: +3.2, +2.6, +1.6, +0.8, +0.3..1, ~0 after. Sum ~ +8.6.
- ORIGINAL MODEL dose_edit (fwd_1, 100 tuning stays sel1 in /work/sel.json): ins x2 -2.01, ins x1/2 +1.16, ne x2 +0.37, ne x1/2 -0.06.
  -> model badly underestimates NE effect; probably underestimates insulin cumulative effect too (check with kernel).
"""
with open('/work/notes.md','a') as f: f.write(notes)
# NE nonlinearity: lag-1 and lag-2 slope by prev-rate bins
for lo,hi in [(0.0,0.1),(0.1,0.2),(0.2,0.35),(0.35,0.5),(0.5,0.7),(0.7,0.85),(0.85,1.01)]:
    s = SN[(SN.ne_prev>lo)&(SN.ne_prev<=hi)]
    a1 = effN(s,'ma_l1'); a2 = effN(s,'ma_l2')
    print(f"prev rate ({lo},{hi}]: n={a1[2]} mean rate {s['ne'].mean():.3f}  lag1 slope {a1[0]:.2f}±{a1[1]:.2f}  lag2 {a2[0]:.2f}±{a2[1]:.2f}")
```