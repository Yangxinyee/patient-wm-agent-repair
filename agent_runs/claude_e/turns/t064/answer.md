Using bundle k+1's posterior z makes my NLL much lower than the tool's, 1.38 against 1.55, and my means worse. So the tool does not use the posterior, and the truth lies between my two versions.

Another explanation could produce an error floor near 3.5: Monte Carlo noise in the tool itself. The tool averages only 4 prior draws per reading. If one z draw moves a predicted mean by several units, the tool's own estimates scatter around the exact expectation. Then no exact copy could get MAE near zero against them.

I will measure three things:
- the spread of the predicted mean across single z draws;
- the MAE between two independent 4-draw estimates from my own head;
- the NLL of 4-draw mixtures compared with the tool's.

If two copies of my own 4-draw estimate also differ by about 3.5, my head is already right and the tool is just noisy. If they agree closely, the mismatch is real.

```python
def p_draws(nd,seed):
    rg=np.random.default_rng(seed); acc=0
    for _ in range(nd): acc=acc+head_p(F,CTX+L(zmu+np.exp(zls)*rg.standard_normal(zmu.shape),'z_head'),1,R,lb,code_ids)
    return acc/nd
singles=np.stack([(p_draws(1,s)[:,1:]*cen).sum(1) for s in range(16)])
print("sd of predicted mean across single z draws: median %.2f mean %.2f"%(np.median(singles.std(0)),singles.std(0).mean()))
pa,pb=p_draws(4,101),p_draws(4,202); ma_,mb_=(pa[:,1:]*cen).sum(1),(pb[:,1:]*cen).sum(1)
print("MAE between two 4-draw estimates: %.3f"%np.abs(ma_-mb_).mean())
print("NLL 4-draw mixtures: %.4f %.4f | tool %.4f"%(-np.log(pa[np.arange(len(F)),bo]).mean(),-np.log(pb[np.arange(len(F)),bo]).mean(),F.nll.mean()))
```