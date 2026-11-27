Run 16 gives the fixed-dose insulin impulse response. Per unit/h it is about −2.0, −2.3, −2.0, −1.6, −1.1, −0.9, −0.6 and −0.4 mg/dL at lags 1–8, summing to about −11 (−12 from lab readings).

The dose-response is close to linear. In tier 2, where most of the variation is, the quadratic term is ≈0, and the 2-hour change falls steadily by about 2.2 mg/dL per unit from rate 1 to 4.5. Tier 1 shows mild extra curvature, but the rates there are small. So ×½ should be roughly −½ of ×2.

Next I'll do the same deconvolution for norepinephrine on arterial MAP:
- total effects T_k of N(t) on MAP(t+k), controlling for the previous rate and current MAP zone;
- persistence C_s of N(t) on N(t+s), expected to be large because titration is multiplicative;
- the implied fixed-dose ψ_k.

I'll also check whether the effect is linear by splitting on rate level. That matters because ×2 pushes rates above the observed cap of 1.0.

```python
for k in range(1,13):
    H[f'ma_p{k}'] = gg.map_arterial.shift(-k); H[f'mc_p{k}'] = gg.map_cuff.shift(-k); H[f'N_p{k}'] = gg.N.shift(-k)
cn = H[(H.N_prev>0)&(H.N>0)&H.map_now.notna()].copy()
cn['z0'] = (cn.map_now<65).astype(float); cn['z2'] = (cn.map_now>=75).astype(float)
cn['Nprev2'] = cn.N_prev**2; cn['Nprev3']=cn.N_prev**3
cn['lNprev'] = np.log(cn.N_prev)
Xn = ['map_now','N_prev','Nprev2','Nprev3','lNprev','z0','z2','N']
Tn={}; Tc={}; Cn={}
for k in range(1,13):
    b,se,n = ols(cn, f'ma_p{k}', Xn); Tn[k]=(b['N'],se['N'])
    b,se,n = ols(cn, f'mc_p{k}', Xn); Tc[k]=(b['N'],se['N'])
    b,se,n = ols(cn, f'N_p{k}', Xn); Cn[k]=(b['N'],se['N'])
Tnk = np.array([Tn[k][0] for k in range(1,13)]); Cnk = np.array([Cn[k][0] for k in range(1,13)]); Tck = np.array([Tc[k][0] for k in range(1,13)])
print(pd.DataFrame({'T_art':{k:f'{v[0]:.2f}±{v[1]:.2f}' for k,v in Tn.items()}, 'T_cuff':{k:f'{v[0]:.2f}±{v[1]:.2f}' for k,v in Tc.items()}, 'C_N':{k:f'{v[0]:.3f}±{v[1]:.3f}' for k,v in Cn.items()}}).to_string())
psi_n = deconv(Tnk, Cnk); psi_c = deconv(Tck, Cnk)
print('psi art ', psi_n.round(2), 'sum', psi_n.sum().round(2))
print('psi cuff', psi_c.round(2), 'sum', psi_c.sum().round(2))
```