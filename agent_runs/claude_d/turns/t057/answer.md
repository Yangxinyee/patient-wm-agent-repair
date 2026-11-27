The clean interior instrument gives a much smaller NE effect. The lag-1 effect is 1.16 per log-unit overall, with a first stage of about 1.0. The low-rate group (mean rate about 0.23) gives 0.93 and the high-rate group (about 0.38) gives 1.42. Per unit of rate that is about 4.0 vs 3.7, nearly constant. So within this range NE now looks linear in rate, and the earlier "log-linear" conclusion was an artefact of the flawed EPSN.

To test linearity over the full 0.08–1.0 range, I'll build a model-based instrument for every titration hour: Z_t = rate_t − E[rate_t | expected rate], computed under the known policy (lognormal noise sd 0.30, cap 1.0, stop below 0.08). That needs exact policy details, so first I'll check the cap value, the minimum running rate, rounding, and a fine-binned f(MAP) from interior rows.

```python
nz = NOR[NOR>0]
print('min nonzero rate', nz.min(), ' frac at 1.0:', np.mean(nz>=0.999), ' max', nz.max())
print('rate quantiles', np.quantile(nz,[0.001,0.01,0.05,0.5,0.95,0.99]).round(3))
print('rounding: unique decimals sample', np.unique(np.round(nz[:2000],4))[:15])
# fine f(MAP) from interior rows
ii,tt=np.where(Sst)
lrr = np.log(np.maximum(NOR[ii,tt],0.07)/NOR[ii,tt-1]); md = Mdec[ii,tt]
for lo_ in range(44,96,4):
    s_=(md>=lo_)&(md<lo_+4)
    if s_.sum()>300: print('Mdec %d-%d n=%5d mean logratio %.3f'%(lo_,lo_+4,s_.sum(),lrr[s_].mean()))
# starts: rate vs MAP at start
st_ = (NOR[:,1:]>0)&(NOR[:,:-1]==0)
sr_ = NOR[:,1:][st_]; sm_ = Mdec[:,1:][st_]
print('starts n', st_.sum(), ' corr(rate,MAP)=%.2f'%np.corrcoef(sr_[~np.isnan(sm_)], sm_[~np.isnan(sm_)])[0,1])
for lo_ in range(40,66,5):
    s_=(sm_>=lo_)&(sm_<lo_+5); print('  start MAP %d-%d n=%d rate mean %.3f sd %.3f'%(lo_,lo_+5,s_.sum(),sr_[s_].mean(),sr_[s_].std()))
```