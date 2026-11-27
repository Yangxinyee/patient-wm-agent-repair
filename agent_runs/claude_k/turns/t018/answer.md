Norepinephrine shows a clear, persistent effect. A +1 mcg/kg/min shock at hour t raises MAP by 3.3 at t+1, 5.4 at t+2 and 6.4 at t+3, then about 5–6 as the rate shock itself decays (0.85, 0.73, 0.64, … through titration). Deconvolving by hand gives a fixed-dose impulse response of roughly 3.3, 2.6, 1.8, 0.9, 0.4, 0.3 mmHg per unit-hour. That is about 9 mmHg at steady state per sustained 1 mcg/kg/min.

One caution: the norepinephrine shock has a larger stay-level share than independent noise would give (0.18 vs about 0.07). The dose scatter could partly track something hidden that also moves MAP or glucose, which would bias these estimates.

Next I'll run balance and placebo checks on both drugs' shocks. Each shock should be uncorrelated with:
- earlier readings (t-1, t-2) given the policy inputs,
- an independent same-hour measurement the policy didn't use (lab glucose when finger was used; cuff MAP when arterial was used),
- covariates like hour and line type.

Correlations near zero would support exogenous shocks and trustworthy estimates. Correlation with the unused same-hour reading would reveal dosing on hidden state, i.e. confounding I must correct.

```python
# insulin shock balance
Lb = L.copy()
jcols = {t: col(t) for t in range(0,25)}
Lb['gl0'] = [gl[s, jcols[t]] for s,t in zip(Lb.stay, Lb.t)]
Lb['gm1'] = [gbest[s, jcols[t]-1] for s,t in zip(Lb.stay, Lb.t)]
Lb['gm2'] = [gbest[s, jcols[t]-2] if t>=1 else np.nan for s,t in zip(Lb.stay, Lb.t)]
s = Lb.dropna(subset=['g0','gl0'])
print("insulin dev ~ lab@t | finger@t, band:", ols(s.dev.values, [s.gl0.values, s.g0.values]+[(s.band==b).values.astype(float) for b in sorted(s.band.unique())[1:]], ['gl0','g0']+[f'b{b}' for b in sorted(s.band.unique())[1:]]).iloc[1:3].round(4).to_dict())
s = Lb.dropna(subset=['g0','gm1','gm2'])
print("insulin dev ~ g(t-1), g(t-2) | g0, band:", ols(s.dev.values, [s.gm1.values, s.gm2.values, s.g0.values, s.t.values]+[(s.band==b).values.astype(float) for b in sorted(s.band.unique())[1:]], ['gm1','gm2','g0','t']+[f'b{b}' for b in sorted(s.band.unique())[1:]]).iloc[1:5].round(4).to_dict())
# norepi shock balance
Nb = N.copy()
Nb['mm1'] = [mlast[s, col(t)-1] for s,t in zip(Nb.stay, Nb.t)]
s = Nb.dropna(subset=['m0','mc0'])
print("norepi ldev ~ cuff@t | art@t, band:", ols(s.ldev.values, [s.mc0.values, s.m0.values]+[(s.mb==b).values.astype(float) for b in sorted(s.mb.unique())[1:]], ['mc0','m0']+[f'b{b}' for b in sorted(s.mb.unique())[1:]]).iloc[1:3].round(5).to_dict())
s = Nb.dropna(subset=['mlast','mm1'])
print("norepi ldev ~ m(t-1), rp, t | mlast, band:", ols(s.ldev.values, [s.mm1.values, s.mlast.values, s.rp.values, s.t.values]+[(s.mb==b).values.astype(float) for b in sorted(s.mb.unique())[1:]], ['mm1','mlast','rp','t']+[f'b{b}' for b in sorted(s.mb.unique())[1:]]).iloc[1:5].round(5).to_dict())
```