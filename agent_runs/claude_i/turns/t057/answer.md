Neither fix helps. A constant flag in any extra column makes the error worse, so the register uses only has_last, log-age and z_last_value. Averaging 16 prior draws is also slightly worse than the prior mean (glucose_finger 1.93 vs 1.86). That is odd if the tool averages random draws, unless the remaining gap is the tool's own 4-draw sampling noise.

Next, I will test that explanation directly. For each reading I'll compute the spread of single-draw predicted means in my replica, which gives the expected error of a 4-draw average. I'll compare that with the observed |tool − mine| and check whether the difference is unbiased and uncorrelated with the inputs. I'll also compare per-reading NLL of the observed bin. If the expected sampling error is about 1.5–2 mg/dL and the residual looks random, the replica is effectively exact. If the residual is systematic, an input is still wrong.

```python
rng_ = np.random.default_rng(1)
singles=[]; Ps=[]
for i in range(16):
    zz = zp_[:,:32] + np.exp(zp_[:,32:])*rng_.standard_normal((len(F),32))
    mn_i,p_i = head_value2(F, out_, m_, gv_, zp_, (), zsamp=zz); singles.append(mn_i); Ps.append(p_i)
singles = np.array(singles); sd1 = singles.std(0)
mn0,p0 = head_value2(F, out_, m_, gv_, zp_)
d_ = y - mn0
for e in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    ix = evs==e
    print(e, "MAD", np.abs(d_[ix]).mean().round(2), "mean diff", d_[ix].mean().round(2), "exp MAD from 4-draw noise ~", (0.8*sd1[ix]/2).mean().round(2),
          "corr(d, mine)", np.corrcoef(d_[ix], mn0[ix])[0,1].round(3))
print("prior logsd mean", zp_[:,32:].mean().round(2), "exp(logsd) mean", np.exp(zp_[:,32:]).mean().round(3))
# NLL comparison
obs = F.observed.values
ob = np.array([fbin(e,'value',np.array([v]))[0] for e,v in zip(evs,obs)])
pav = np.mean(Ps,0)
nll_mine = -np.log(pav[np.arange(len(F)), ob]); nll_mean = -np.log(p0[np.arange(len(F)), ob])
print("NLL tool", F.nll.mean().round(4), "mine(avg16)", nll_mine.mean().round(4), "mine(prior mean)", nll_mean.mean().round(4))
```