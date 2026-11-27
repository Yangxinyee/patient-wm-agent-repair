The caches include original-logit factual anchors and paired dose contrasts, so further candidates can be fitted analytically. I’ll compare the existing dose decoders and regression weights. A better decoder should preserve the physical dose contrast while varying less on factual records; otherwise stronger causal corrections are likely to cost too much factual fit.

```python
print("=== CURRENT ANALYTIC FIT ===")
for k259, v259 in st242.items():
    if isinstance(v259, np.ndarray):
        print(k259, v259.shape, "RMS", float(np.sqrt(np.mean(v259**2))))
    elif isinstance(v259, dict):
        print(k259, "keys", list(v259))
        if k259 in ("GD", "RD"):
            for kk259, vv259 in v259.items():
                print(" ", kk259, vv259.shape, "RMS", float(np.sqrt(np.mean(vv259**2))))
    else:
        print(k259, v259)

print("\n=== EXISTING GRID SCORES ===")
for k259, v259 in grid242.items():
    print(k259, v259["score"])

print("\n=== DOSE DECODER GEOMETRY ===")
for drug259 in ("insulin", "norepinephrine"):
    p259 = paired214[(drug259, 2.0)]
    own259 = family204[drug259]
    for lam259 in (0.001, 0.01, 0.03, 0.1, 0.3, 1.0):
        d259 = directions207[(drug259, lam259)]
        f259 = Xraw204[:, own259] @ d259
        pred259 = p259["dx"][:, p259["own"]] @ d259
        truth259 = p259["dr"][p259["own"]][None, :]
        print(drug259, lam259,
              "norm", round(float(np.linalg.norm(d259)), 3),
              "factual_SD", round(float(f259.std()), 4),
              "contrast_RMSE", round(float(np.sqrt(np.mean((pred259-truth259)**2))), 4),
              "contrast_slope", round(float(np.mean(pred259*truth259)/np.mean(truth259**2)), 4))
    print("center norm:", float(np.linalg.norm(dosecenter214[drug259])))

print("\n=== RAW TUNING CACHE NAMES ===")
for n259, v259 in sorted(list(globals().items())):
    if isinstance(v259, np.ndarray) and (
        (re.search(r"(225|226|227|228|229|230|231|232|233|234)$", n259)
         and (n259.startswith(("X", "F", "H", "L", "C", "B"))))
        or "tune" in n259.lower()
    ):
        print(n259, v259.shape)

print("\n=== FEATURE CONSTRUCTION, FINAL PART ===")
for i259 in dis.get_instructions(make_arch215):
    if i259.offset >= 690:
        print(i259.opname, i259.argrepr)
```
