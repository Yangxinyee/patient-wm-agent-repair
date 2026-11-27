Run 13 indicates that norepinephrine titration noise is multiplicative and close to exogenous:
- The SD of the log-ratio is about 0.2–0.5, and the SD of dN grows with the rate.
- The residual is essentially uncorrelated with the MAP level within each zone and only weakly with the MAP trend (−0.04).
- The lag-1 correlation of −0.12 is plausibly an artefact of the cap at 1.0 (rate = min(1, prev·exp(a_zone+ε))) and of my crude residual.

So the norepinephrine rate given (previous rate, current MAP) is effectively randomised, like insulin.

Next I'll estimate the norepinephrine effect on later MAP the same way. I'll regress MAP at t+k (arterial and cuff separately) on N(t), controlling for N(t−1) and current MAP, among rows on norepinephrine. A clearly positive coefficient means norepinephrine raises MAP. A zero would mean no effect, whatever medicine says.

```python
for k in range(1,5):
    H[f'ma_p{k}'] = gg.map_arterial.shift(-k); H[f'mc_p{k}'] = gg.map_cuff.shift(-k)
cn = H[(H.N_prev>0)&(H.N>0)&H.map_now.notna()].copy()
cn['z0'] = (cn.map_now<65).astype(float); cn['z2'] = (cn.map_now>=75).astype(float)
cn['Nprev2'] = cn.N_prev**2
for k in [1,2,3,4]:
    for y in [f'ma_p{k}', f'mc_p{k}']:
        b,se,n = ols(cn, y, ['map_now','N_prev','Nprev2','z0','z2','N'])
        print(f'lag {k} {y}: coef N {b["N"]:.2f} (se {se["N"]:.2f}) coef MAP {b["map_now"]:.3f}  coef Nprev {b["N_prev"]:.2f} n={n}')
```