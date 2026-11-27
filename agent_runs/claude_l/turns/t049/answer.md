The direct-effect regression changes the picture. The true one-step conditional depends strongly on doses two or more hours back, even holding the recent readings fixed:
- Insulin on the next finger glucose, by lag: −1.96, −1.32, −0.52 for lags 1–3, then a positive tail of about +0.3 to +0.7 over lags 4–10. The tail is the glucose recovery once the recent drop is known to be drug-caused.
- Norepinephrine on the next arterial MAP: +3.2 and +3.6 at lags 1–2, with a smaller positive tail after.

The model's view carries none of the dose residuals at lag ≥2, so it cannot represent these terms. That explains why its rollouts are 4 to 8 times too weak.

A correction at the head alone cannot fix this. The core state needs a dose memory, and designing one that is controlled and does not harm the factual fit needs an exact local reimplementation of the model. With one I can run teacher-forced passes and dose-edit rollouts on the CPU and test edits for free, keeping forward passes for validation.

My earlier attempt stalled at a mismatch of about 0.46. So I will rebuild the input side carefully, starting from `vocab.json`: codes, kinds, the categorical tables, the role of `embed.kq` and `embed.text`, the z-score normalisation and the bin edges. First I save the new findings to the notebook.

```python
with open('/work/notes.md', 'a') as f_:
    f_.write("""
## Restart 1 (runs 39-46)
- Value head = Linear192-act-Linear17 on with_code (q=query_norm(view+type_emb)...). View decodes ins dose-residual lag1 only (CV R2 .30), lags>=2 R2~0; NE lag1 .05. => model has NO dose-residual memory beyond ~1h.
- True DIRECT conditional coefs (G_t on last readings, pihat lags, eps lags; train, OLS):
  ins->finger eps lag1..12: -1.96 -1.32 -.52 +.31 +.52 +.65 +.69 +.40 +.50 +.57 +.13 +.19 (se ~.07)
  NE->art eps lag1..6: +3.23 +3.59 -.32 +.93 +1.23 +.42 (se .1-.3). Positive tails = recovery given readings.
- Policy GBM (pcols feats from feats()): ins tuning R2 .87 resid sd .35; NE R2 .94 sd .073.
- Conclusion: head-only correction cannot work (info absent from view). Need core-level dose memory -> need exact reimplementation.
""")
V = VOCAB
print(list(V.keys()))
for k_ in V:
    if k_ in ('codes',):
        print(k_, V[k_])
    else:
        s_ = json.dumps(V[k_]); print(k_, len(s_), s_[:700])
```