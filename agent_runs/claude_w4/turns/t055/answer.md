Run 50 settles one question. The state just before each reading encodes the latest glucose almost perfectly (R² 0.95–0.98) and recent insulin rates moderately (R² 0.6–0.8). It does not encode the dose surprises at all (R² ≈ 0). So the rate information the state keeps is only the part predictable from the glucose band. The within-band variation, the part carrying the causal effect, is discarded.

A heads-only edit therefore cannot work. The dose surprise has to be written into the state by a core-level edit.

The z posterior is the one place where context and action meet linearly: `z_post` sees [ctx, e, u]. It is trained to encode each bundle's unexpected content. At an insulin bundle the unexpected content is exactly the rate surprise.

If z at insulin bundles already encodes the surprise, the cleanest edit routes that z-direction onto the state's glucose belief. That would avoid computing the expected dose myself. I will regress z at insulin bundles on the insulin surprise and the expected rate, and do the same for norepinephrine bundles (rate change from the previous hour). I will save my notes first.

- High R² for the surprise means the information is already computed in z.
- R² near 0 means I must compute the surprise in `z_post` myself.

```python
notes_add = """
## Runs 37-50 (after restart)
- fwd_1: ORIGINAL dose_edit (tuning sel100): ins x2 -0.348, ins x1/2 +0.208, NE x2 +0.018, NE x1/2 -0.245 -> model ~ignores doses.
- fwd_2: ORIGINAL factual sel100: value NLL 1.49383 (finger 1.5515, lab 1.2396, art 1.3522, cuff 1.6918). fwd_3: original states sel100.
- Vocab: padding rows (index 0) of bins/categorical are zero. predicted mean = sum p(b)*bin centre (16 coarse bins, top finger bin >258 centre 269).
  insulin rate z=(r-1.5)/1, bins edges .5,.5,.5,1,1,1,1,1.5,1.5,2,2.5,3,3.5,4,5 ; NE rate z=(r-.47)/.6.
- FACTUAL RESIDUAL TEST: regress (observed - old predicted mean) on insulin surprise s=r-E[r|band] lags1-8 (band_mean: 0,1.06,2.10,3.15,4.28):
  finger gamma=[-2.05,-1.64,-1.33,1.11,-.09,.34,.27,-.52] se~.6-.7 ; lab [-1.32,-.79,-1.22,...] se~1.2. -> old model ignores dose surprise; adding it improves factual fit.
- Tried to replicate model on CPU (runs 44-49): could not match z posterior (mean err ~.04 vs cross-stay sd .07). ABANDONED replica.
- States (fwd_3) ridge CV R2: out->latest glucose .95-.98, ins rate lag1 .76 lag2 .71, NE rate .46-.57, but insulin SURPRISE s1,s2 R2~0.
  -> state keeps only band-predictable dose info; heads-only edit impossible; must inject surprise into core.
"""
with open('/work/notes.md','a') as f_: f_.write(notes_add)
iz = np.flatnonzero(S['has_insulin']); nz = np.flatnonzero(S['has_norepinephrine'])
def lagkey(col, i, j=0): return Tser[col].get((sid_[i], int(np.floor(hr_r[i])) - j), np.nan)
Zall = X_[:, 960:992]
s_at = np.array([lagkey('s_ins', i) for i in iz]); r_at = np.array([lagkey('ins_rate', i) for i in iz]); rh_at = r_at - s_at
for nm_, yv in [('s', s_at), ('rhat', rh_at), ('r', r_at)]:
    okm = ~np.isnan(yv); pred = np.zeros(okm.sum())
    for trn, tst in GroupKFold(5).split(Zall[iz][okm], yv[okm], sid_[iz][okm]):
        pred[tst] = Ridge(alpha=1.0).fit(Zall[iz][okm][trn], yv[okm][trn]).predict(Zall[iz][okm][tst])
    print(f'insulin bundles: z -> {nm_}: n={okm.sum()} CV R2={1-np.mean((yv[okm]-pred)**2)/np.var(yv[okm]):.3f}')
ne_r = np.array([lagkey('ne_rate', i) for i in nz]); ne_p = np.array([lagkey('ne_rate', i, 1) for i in nz])
dl = np.log(ne_r) - np.log(ne_p); okm = ~np.isnan(dl)
pred = np.zeros(okm.sum())
for trn, tst in GroupKFold(5).split(Zall[nz][okm], dl[okm], sid_[nz][okm]):
    pred[tst] = Ridge(alpha=1.0).fit(Zall[nz][okm][trn], dl[okm][trn]).predict(Zall[nz][okm][tst])
print(f'NE bundles: z -> dlog rate: n={okm.sum()} CV R2={1-np.mean((dl[okm]-pred)**2)/np.var(dl[okm]):.3f}')
```