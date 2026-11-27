Run 14 shows that norepinephrine truly raises MAP. Using the randomised titration variation, controlling for the previous rate and current MAP, the effect per 1 mcg/kg/min is +3.4 mmHg at lag 1, about +5.1 at lag 2, +5.6 at lag 3 and +5.2 at lag 4. Arterial and cuff readings agree.

These are total effects. Because the titration is multiplicative, a higher rate now persists into later rates, so the lagged coefficients include the effect of a raised future rate path. The controlled one-hour effect needed for a fixed-dose edit will be smaller and must be deconvolved.

There is also a confounding signal: holding the current rate and MAP fixed, a higher previous rate predicts *lower* future MAP (coefficient −7.6). The previous rate is a proxy for latent shock severity. This is exactly the association a model could misread as a drug effect.

Next I'll do the deconvolution for insulin. I'll estimate total effects T_k of I(t) on G(t+k) (k=1..10) and feedback C_s of I(t) on later insulin I(t+s), both from within-tier variation. I'll also check whether finger-reading availability depends on dose, which would cause selection bias. Then I'll solve T_k = Σ_s C_s ψ_{k−s} for the fixed-dose impulse response ψ.

```python
gg = H.groupby('stay_id')
for k in range(1,13):
    H[f'gf_p{k}'] = gg.glucose_finger.shift(-k); H[f'gl_p{k}'] = gg.glucose_lab.shift(-k); H[f'I_p{k}'] = gg.I.shift(-k)
    H[f'obsf_p{k}'] = H[f'gf_p{k}'].notna().astype(float)
H['gf_m1'] = gg.glucose_finger.shift(1)
base_ins = H[(H.tier>=1)&(H.I>0)].copy()
for t_ in [1,2,3,4]: base_ins[f'T{t_}'] = (base_ins.tier==t_).astype(float)
X = ['glucose_finger','T2','T3','T4','I']
T = {}; Tl={}; C = {}; O={}
for k in range(1,13):
    b,se,n = ols(base_ins, f'gf_p{k}', X); T[k]=(b['I'],se['I'])
    b,se,n = ols(base_ins, f'gl_p{k}', X); Tl[k]=(b['I'],se['I'])
    b,se,n = ols(base_ins, f'I_p{k}', X); C[k]=(b['I'],se['I'])
    b,se,n = ols(base_ins, f'obsf_p{k}', X); O[k]=(b['I'],se['I'])
print(pd.DataFrame({'T_finger':{k:f'{v[0]:.2f}±{v[1]:.2f}' for k,v in T.items()}, 'T_lab':{k:f'{v[0]:.2f}±{v[1]:.2f}' for k,v in Tl.items()},
                    'C_dose':{k:f'{v[0]:.3f}±{v[1]:.3f}' for k,v in C.items()}, 'obs_f':{k:f'{v[0]:.4f}±{v[1]:.4f}' for k,v in O.items()}}).to_string())
```