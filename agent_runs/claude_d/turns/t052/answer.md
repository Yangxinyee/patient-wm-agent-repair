The log-rate IV gives γ = [1.76, 1.50, 1.00, 0.51, 0.07, …] mmHg per log-unit at lags 1–8. But the per-quartile lag-1 regression in run 46 gave about 3.1–3.5. Both should estimate the same lag-1 effect.

The main difference is that run 46 controlled for the current MAP (MA_t), which the policy reacts to. If my noise residual EPSN is not pure noise, the uncontrolled IV is biased. That would happen if the real titration factor depends on MAP more smoothly than my step function f(MAP), leaving part of MAP inside EPSN.

I'll check whether EPSN_t is correlated with MA_t, MA_{t−1} and the previous rate, and print the titration pattern by MAP band. A clear correlation would mean EPSN needs re-cleaning before it can serve as an instrument.

```python
ok = (NOR[:,1:25]>0)&(NOR[:,0:24]>0)&(EPSN[:,1:25]!=0)&~np.isnan(MA[:,1:25])
e_ = EPSN[:,1:25][ok]; m_ = MA[:,1:25][ok]; r_ = NOR[:,0:24][ok]; mm1 = MA[:,0:24][ok]
lr_ = np.log(NOR[:,1:25][ok]/NOR[:,0:24][ok])
print('n',ok.sum(),'corr(EPSN, MAP_t)=%.3f corr(EPSN, rate_{t-1})=%.3f corr(EPSN,MAP_{t-1})=%.3f'%(np.corrcoef(e_,m_)[0,1], np.corrcoef(e_,r_)[0,1], np.nanmean(np.corrcoef(e_[~np.isnan(mm1)],mm1[~np.isnan(mm1)])[0,1])))
# titration log-ratio vs MAP bins
mb_ = np.arange(50,96,3)
for lo_ in mb_:
    s_=(m_>=lo_)&(m_<lo_+3)
    if s_.sum()>200: print('MAP %d-%d n=%6d mean logratio %.3f  sd %.3f  mean EPSN %.3f'%(lo_,lo_+3,s_.sum(),lr_[s_].mean(),lr_[s_].std(),e_[s_].mean()))
```