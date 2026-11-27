The z-draw noise is negligible: only 0.2% of the pair output's variance is across draws, and a single draw correlates with the true deviation as well as the 16-draw mean (0.70). So the readout noise comes from how the state encodes the dose, not from sampling.

Where things stand:
- NE: its readout is too noisy (R² 0.19), leaks onto map_arterial rows without NE, and cannot fix ×2. Under titration, a dose-deviation readout also stops firing after the first doubled rate. I'll drop the NE part and restore those two units.
- Insulin: the budget allows roughly a = −2.4 to −2.6 (about 0.0034–0.0039 per reading on a random-stay mix). Linear scaling gives insulin ×2 ≈ −12 to −12.8 against truth −13.4.

One possible source of readout error is fixable. The readout must subtract the sliding-scale expected dose, a step function of the last finger reading with steps at 150, 180, 220 and 300. If those thresholds fall inside the value bins, a linear readout cannot reproduce the steps. A GELU unit keyed to the exact last value could.

To test this, I'll print the finger-glucose bin edges. Then I'll look at the readout error (pair output minus gain × true deviation) by how close the last reading was to a threshold. A large error near thresholds would confirm the idea; a flat error would rule it out.

```python
vb = VOCAB.get('value_bin_edges', None) if isinstance(VOCAB, dict) else None
print('VOCAB keys:', list(VOCAB.keys())[:20])
for k in VOCAB.keys():
    if 'edge' in k.lower():
        e_ = VOCAB[k]
        print(k, type(e_), (list(e_.keys())[:10] if isinstance(e_, dict) else None))
        if isinstance(e_, dict):
            for kk in e_:
                if 'glucose_finger' in str(kk) or str(kk) in ('6','glucose_finger'):
                    print('  gf edges', np.round(np.asarray(e_[kk], float), 1))
s_ = R1[(R1.code==6) & R1.drug].copy()
s_['r'] = mean_r
gain_r = np.polyfit(s_.dev1t, s_.r, 1)[0]
s_['err'] = s_.r - gain_r * s_.dev1t
s_['g_prev'] = g_prev[((R1.code==6) & R1.drug).values]
thr = np.array([150,180,220,300.])
s_['dthr'] = np.min(np.abs(s_.g_prev.values[:,None] - thr[None,:]), axis=1)
s_['near'] = pd.cut(s_.dthr, [-1, 5, 10, 20, 1000])
print('gain', round(gain_r,3), 'err sd', round(s_.err.std(),3))
print(s_.groupby('near').err.agg(['size','mean','std']).round(3).to_string())
print(s_.groupby(tier5(s_.g_prev.values)).err.agg(['size','mean','std']).round(3).to_string())
```