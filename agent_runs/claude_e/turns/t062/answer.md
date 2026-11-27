Every activation, gap scaling and prior ordering plateaus at MAE 3.47. So the mismatch is not in those choices. It comes from something the formula does not show, or from how the reported mean is formed.

Candidates:
- **Renormalisation.** The tool may divide by the probability of having a value (class 0 removed), while I use the plain sum. That would make me systematically low, which fits the −1.7 and −2.9 glucose biases.
- **Other patterns.** Errors may depend on how many readings share the bundle (sequential prediction within a bundle), on the hour, or on the gap.

I will compare both mean definitions, my bin NLL against the tool's, and the error split by bundle size. If renormalising sharply reduces the MAE, the class-0 handling was the bug.

```python
p_=head_p(F,Vm,1,R,lb,code_ids); p0=p_[:,0]
m_plain=(p_[:,1:]*cen).sum(1); m_ren=m_plain/p_[:,1:].sum(1)
print("p(class0): mean %.3f max %.3f"%(p0.mean(),p0.max()))
print("MAE plain %.3f | renorm %.3f"%(np.abs(m_plain-tool).mean(),np.abs(m_ren-tool).mean()))
bo=np.array([fbin(ev,np.array([v]))[0] for ev,v in zip(F.event,F.observed)])
nll_m=-np.log(p_[np.arange(len(F)),bo]); nll_r=-np.log(p_[np.arange(len(F)),bo]/p_[:,1:].sum(1))
print("NLL tool %.4f | mine plain %.4f | mine renorm %.4f | corr %.3f"%(F.nll.mean(),nll_m.mean(),nll_r.mean(),np.corrcoef(nll_m,F.nll)[0,1]))
F['nb']=F.groupby(['stay_id','hour']).event.transform('count'); e_=np.abs(m_plain-tool)
print("MAE by bundle size:",{int(k):round(e_[(F.nb==k).values].mean(),2) for k in sorted(F.nb.unique())})
```