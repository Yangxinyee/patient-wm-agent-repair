The linear-pair restriction fails the independent factual check, so I’m rejecting it. Protecting the older features gives only a small improvement. Higher-regularization dose decoders have lower factual variance while retaining most of the dose contrast; I’ll rebuild their anchored regressions and test whether that buys room for stronger effects.

```python
print("=== FACTUAL NORMALIZATION CHECK ===")
fa261 = st242["arch"]["F"][:, fitmask179].reshape(-1, 193)
fq261 = st242["Ftune"].reshape(-1, 193)
sf261 = np.sqrt(np.mean(fa261**2, axis=0))
sq261 = np.sqrt(np.mean(fq261**2, axis=0))
sa261 = np.sqrt((np.sum(fa261**2, axis=0)+np.sum(fq261**2, axis=0))
                /(len(fa261)+len(fq261)))
print("row counts:", len(fa261), len(fq261))
for label261, ss261 in (("train", sf261), ("tuning", sq261), ("combined", sa261)):
    print(label261, "scale max error",
          float(np.max(np.abs(ss261-st242["scale"]))))
for wt261 in (0.0, 0.5, 1.0, 2.0):
    gf261 = ((fa261.T@fa261)/len(fa261) + wt261*(fq261.T@fq261)/len(fq261))/(1+wt261)
    gf261 /= st242["scale"][:, None]*st242["scale"][None, :]
    print("tuning weight", wt261, "GF max error",
          float(np.max(np.abs(gf261-st242["GF"]))))

print("\n=== CONTRAST REGRESSION IMPLEMENTATION ===")
for i261 in dis.get_instructions(fit_arch215):
    print(i261.opname, i261.argrepr)

print("\n=== HIGHER-REGULARIZATION FACTUAL VARIANCE ===")
familyq261 = np.mean(X225@gatecanon200, axis=0) > 0
for drug261 in ("insulin", "norepinephrine"):
    mq261 = familyq261 if drug261 == "insulin" else ~familyq261
    for lam261 in (0.1, 0.3, 1.0):
        d261 = directions207[(drug261, lam261)]
        z261 = X225[:, mq261]@d261
        print(drug261, lam261, "independent mean/SD",
              float(z261.mean()), float(z261.std()))
```
