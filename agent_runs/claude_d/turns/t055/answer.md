The NE policy decides on map_arterial when it is present and on map_cuff otherwise (Mdec). Titration multiplies the rate by about e^{+0.15..0.18} below MAP 65, by 1 between 65 and 75, and by e^{−0.16} above 75. The noise sd is about 0.30.

Next, the clean instrument. Z_t is the log rate ratio, residualised on fine bins of Mdec_t and of log rate_{t−1}. It is used only where the pre-decision expected rate lies in [0.18, 0.45], so capping or stopping happens under 0.5% of the time. This selection uses pre-treatment variables only, so it adds no bias.

I'll then fit the IV impulse response under three dose codings: log, linear and Emax. To choose between them, I'll re-estimate the lag-1 and lag-2 effects separately with instruments from low and from high previous-rate hours. The right coding should give equal effects in the two groups.

```python
def f_ne(m): return np.where(m<65, np.log(1.15), np.where(m<75, 0.0, np.log(0.85)))
Nn, Hn = NOR.shape
Zc = np.zeros_like(NOR); Sst = np.zeros_like(NOR, bool)
exp_rate = np.zeros_like(NOR)
for t in range(1,Hn):
    pre = (NOR[:,t-1]>0)&~np.isnan(Mdec[:,t])
    er = NOR[:,t-1]*np.exp(f_ne(np.nan_to_num(Mdec[:,t],nan=70)))
    s_ = pre&(er>=0.18)&(er<=0.45)
    Sst[:,t]=s_; exp_rate[:,t]=er
    Zc[s_,t] = np.log(np.maximum(NOR[s_,t],0.07)/NOR[s_,t-1])
# residualize within S on bins of Mdec and log prev rate
ii,tt = np.where(Sst)
md_b = np.clip(np.round(Mdec[ii,tt]),40,100).astype(int); lr_b = np.round(np.log(NOR[ii,tt-1])*10).astype(int)
dfz = pd.DataFrame({'z':Zc[ii,tt],'m':md_b,'r':lr_b})
dfz['z'] = dfz['z'] - dfz.groupby('m')['z'].transform('mean')
dfz['z'] = dfz['z'] - dfz.groupby('r')['z'].transform('mean')
Zc[ii,tt] = dfz['z'].values
print('n instrument rows', len(ii), 'sd', dfz['z'].std().round(3))
codings = {'log': np.maximum(np.where(NOR>0, np.log(np.maximum(NOR,1e-3)/0.08),0),0), 'lin': NOR.copy()}
for ec in [0.1,0.3]: codings['emax%.1f'%ec] = NOR/(ec+NOR)
K=8
for nm_, Ac in codings.items():
    g_, se_, n_ = iv_fit(MA, Ac, Zc, K)
    print('%-8s'%nm_, 'gamma', g_.round(2), ' se1', se_[0].round(2))
    # heterogeneity: instruments only from low / high prev-rate strata
    lowmask = Sst & (np.vstack([np.zeros(Nn), NOR[:,:-1].T]).T < 0.3)
    out_=[]
    for msk_ in [lowmask, Sst & ~lowmask]:
        g2,se2,_ = iv_fit(MA, Ac, np.where(msk_, Zc, 0.0), K)
        out_.append('%.2f(%.2f) %.2f'%(g2[0],se2[0],g2[1]))
    print('       low-rate inst:', out_[0], ' high-rate inst:', out_[1])
```