Forward pass 2 gives the original model's teacher-forced value NLL on these 100 stays: 1.404 nats per reading over 5,802 values. By event: glucose_finger 1.530 (MAE 10.6), glucose_lab 1.216, map_arterial 1.300 (MAE 4.1) and map_cuff 1.534.

Next I'll join the per-reading predicted means to my hourly panel. Then I'll measure the model's one-step dose sensitivity, comparing it with the same regression on the actual readings:
- for glucose, regress the predicted mean finger glucose at t+1 on I(t) within tier, controlling for G(t);
- for MAP, regress the predicted arterial MAP at t+1 on N(t) with the previous rate and MAP controls.

The model's coefficient against the truth (−2 and +3.4) tells me whether the one-step causal response is already right.

```python
f2 = pd.read_csv('/work/fwd_2/factual_original.csv')
print(f2.shape, f2.columns.tolist()); print(f2.head(8).to_string())
f2['h'] = np.floor(f2.hour+1e-9).astype(int) if 'hour' in f2.columns else None
pm = f2.pivot_table(index=['stay_id','h'], columns='event', values=[c for c in f2.columns if c.startswith('pred')][0], aggfunc='first')
pm.columns = ['pred_'+c for c in pm.columns]
Hs = H[H.stay_id.isin(sel2)].merge(pm.reset_index(), on=['stay_id','h'], how='left')
gs = Hs.groupby('stay_id')
for c in ['pred_glucose_finger','pred_map_arterial']:
    for k in [1,2,3]:
        Hs[f'{c}_p{k}'] = gs[c].shift(-k)
bi = Hs[(Hs.tier>=1)&(Hs.I>0)].copy()
for t_ in [2,3,4]: bi[f'T{t_}'] = (bi.tier==t_).astype(float)
for k in [1,2,3]:
    b1,se1,n1 = ols(bi, f'gf_p{k}', ['glucose_finger','T2','T3','T4','I'])
    b2,se2,n2 = ols(bi, f'pred_glucose_finger_p{k}', ['glucose_finger','T2','T3','T4','I'])
    print(f'lag{k} glucose: actual coef I {b1["I"]:.2f}±{se1["I"]:.2f} (n={n1}) | model pred coef I {b2["I"]:.2f}±{se2["I"]:.2f} (n={n2})')
bn = Hs[(Hs.N_prev>0)&(Hs.N>0)&Hs.map_now.notna()].copy()
bn['z0'] = (bn.map_now<65).astype(float); bn['z2'] = (bn.map_now>=75).astype(float); bn['Nprev2']=bn.N_prev**2; bn['lNprev']=np.log(bn.N_prev)
for k in [1,2,3]:
    b1,se1,n1 = ols(bn, f'ma_p{k}', ['map_now','N_prev','Nprev2','lNprev','z0','z2','N'])
    b2,se2,n2 = ols(bn, f'pred_map_arterial_p{k}', ['map_now','N_prev','Nprev2','lNprev','z0','z2','N'])
    print(f'lag{k} MAP: actual coef N {b1["N"]:.2f}±{se1["N"]:.2f} (n={n1}) | model pred coef N {b2["N"]:.2f}±{se2["N"]:.2f} (n={n2})')
```