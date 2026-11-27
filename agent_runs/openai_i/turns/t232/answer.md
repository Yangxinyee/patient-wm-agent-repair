The caches retain gated dose innovations and original latent-token targets. I’ll test whether those innovations can occupy redundant latent coordinates while leaving attention unchanged. A small held-stay token error would justify testing that simpler route; a large error would rule it out.

```python
Z207 = np.asarray(all_z201, dtype=np.float64)
Y207 = np.asarray(Yzin196, dtype=np.float64)
innovation207 = np.asarray(feature_pred201, dtype=np.float64)
print("Target token RMS:", np.sqrt(np.mean(Y207**2)))
print("Original target consistency:",
      np.sqrt(np.mean((np.asarray(Zbase196) @ asnp175(load_original("z_in.weight")).T
                       + asnp175(load_original("z_in.bias")) - Y207)**2)))

corr207 = np.corrcoef(np.c_[Z207, innovation207], rowvar=False)[:32, 32:]
private_indices207 = np.argmax(np.abs(corr207), axis=0)
print("Innovation coordinates:", private_indices207)
print("Their correlations:", corr207[private_indices207, np.arange(2)])
print("Coordinate changes:",
      np.round(np.sqrt(np.mean((Z207 - np.asarray(Zbase196))**2, axis=0)), 5))

fit_token207 = np.zeros(len(Z207), dtype=bool)
fit_token207[np.asarray(source201)[np.asarray(fit_att201).reshape(-1, 4).all(axis=1)]] = True
kept_z207 = np.setdiff1d(np.arange(32), private_indices207)
X207 = np.c_[Z207[:, kept_z207], np.ones(len(Z207))]
reg207 = np.eye(X207.shape[1]) * 1e-7
reg207[-1, -1] = 1e-10
coef_token207 = np.linalg.solve(
    X207[fit_token207].T @ X207[fit_token207] + reg207,
    X207[fit_token207].T @ Y207[fit_token207]
)
pred_token207 = X207 @ coef_token207
print("Token reconstruction train/held RMS:",
      np.sqrt(np.mean((pred_token207[fit_token207] - Y207[fit_token207])**2)),
      np.sqrt(np.mean((pred_token207[~fit_token207] - Y207[~fit_token207])**2)))
print("Held token absolute error quantiles:",
      np.quantile(np.abs(pred_token207[~fit_token207] - Y207[~fit_token207]),
                  [.5, .9, .99, 1]))
print("Fit/held rows:", int(fit_token207.sum()), int((~fit_token207).sum()))

zin_neutral207 = np.zeros((192, 32), dtype=np.float32)
zin_neutral207[:, kept_z207] = coef_token207[:-1].T.astype(np.float32)
zin_bias207 = coef_token207[-1].astype(np.float32)

print("\nFUNCTION CONTRACTS:")
for fn207 in [core_step136, batch_replay182, batch_q182, o191, attention_features196]:
    print(fn207.__name__, "names:", fn207.__code__.co_names)
    print("constants:", [x for x in fn207.__code__.co_consts
                         if isinstance(x, (str, int, float))][:20])
print("\nWEIGHT CACHE NAMES:",
      [k for k, v in list(globals().items())
       if isinstance(v, dict) and "z_in.weight" in v])
print("ACTIVE EDITS:", edited())
```