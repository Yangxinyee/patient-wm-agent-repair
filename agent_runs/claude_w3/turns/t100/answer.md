Removing the time flow while keeping `gap(log dt / 10)` with dt in seconds is the best combination, at RMSE 0.081. Removing `gap` makes it worse. Zeroing e_k doubles the error, so e_k matters and is probably where the remaining error sits.

Next I will redo the search over the measurement-unit mapping and the two bin rules with the corrected context. A clear winner near zero error would settle the measurement embedding. Errors still near 0.08 for all mappings would point to another embedding term I have missed.

```python
ctx_nf = v2['nf_gap_s1p'] if False else (lambda p, d: p[:192] + HN(p, d, False) + G_(d))
out_ = []
for rule in ['lt','le']:
    fb = fb_lt if rule=='lt' else (lambda c_, a_, x: int(1 + np.sum(NUM[(c_, a_)][2] <= x)))
    for mg in [2,3,4,5]:
        for mm in [2,3,4,5]:
            if mg==mm: continue
            CFG['unit'] = {'glucose_finger': mg, 'glucose_lab': mg, 'map_arterial': mm, 'map_cuff': mm}
            out_.append((zerr(ctx_nf), rule, mg, mm))
fb = fb_lt
out_.sort(); print(out_[:5], out_[-1])
```