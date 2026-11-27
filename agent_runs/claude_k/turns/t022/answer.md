Partialling-out confirms insulin. A dose shock at t moves glucose_finger by -1.99, -2.27, -1.84, -1.33, -0.78, -0.47, -0.27, -0.17 at lags 1–8, with SE about 0.05–0.08. Later doses shift only by -0.05 to -0.07.

For norepinephrine, the first lag is unchanged at +3.33, but later lags are smaller than the crude estimates: 5.07, 5.55, 5.26, 4.52, 3.86, 3.11, 2.65. The shock's persistence in later rates is also lower (0.74, 0.56, 0.44, …). Dropping capped rows had inflated both.

Next I'll deconvolve both into fixed-dose impulse responses h_k with h_k = total_k − Σ_{j≥1} h_{k−j}·f_j. Then I'll check the standard errors by bootstrapping over stays. A positive norepinephrine h at every lag that decays like insulin's would confirm a short-memory linear effect in both drugs. Negative or oscillating late values would point to estimation noise or a rebound I need to model.

```python
def deconv(tot, f):
    h=[]
    for k in range(1,len(tot)+1):
        s = tot[k-1] - sum(h[k-1-j]*f[j-1] for j in range(1,k))
        h.append(s)
    return np.array(h)
for drug in ['ins','ne']:
    tot = [o[2] for o in results[drug]]; f = [o[4] for o in results[drug]]
    h = deconv(tot, f)
    print(drug, "impulse h_k:", np.round(h,3), " cumsum:", np.round(np.cumsum(h),2))
```