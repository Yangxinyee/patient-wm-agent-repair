The prelude-FFN route looks weak so far:
- The context decoders work (correlation 0.86 for insulin rate, 0.80 for NE).
- A linear read-out of the FFN input recovers the insulin innovation with cross-validated R² of only 0.47, and the NE innovation with 0.22. Errors that large would inject noise costing far more than the 0.005-nat budget.
- The drug-vs-reading gate is mostly separable but has outliers.

A likely reason is that the dose arrives in the slots only through attention to the u token, with a weight that varies by slot and bundle. A linear unit can't undo that multiplicative factor, and the small dose component may also have been ridge-suppressed. To see which part fails, I'll decode the dose alone (multiplier × rate) and the context expectation alone, separately per slot and with much weaker regularisation. If the dose part decodes well (R² > 0.95) and only the context part fails, a two-part design remains possible. If the dose part also fails, this route is closed.

```python
def r2cv(X,y,fold,al):
    p=ridge_cv(X,y,fold,al)[1]; return 1-((y-p)**2).mean()/y.var()
yD=np.repeat((mAv[:,None]*ri0[None]).reshape(-1),4); yE=np.repeat(np.tile(EA,len(VAR)),4)
selA=np.repeat(np.repeat((mBv==1)[:,None],nd,1).reshape(-1),4)
for al in [1e-2,1e-5]:
    print('al',al,'dose R2 %.3f  Ehat R2 %.3f'%(r2cv(Xd[selA],yD[selA],fd[selA],al), r2cv(Xd[selA],yE[selA],fd[selA],al)))
for s in range(4):
    m_=selA&(np.tile(np.arange(4),len(Xd)//4)==s)
    print('slot',s,'dose R2 %.3f innovA R2 %.3f'%(r2cv(Xd[m_],yD[m_],fd[m_],1e-5), r2cv(Xd[m_],yA[m_],fd[m_],1e-5)))
```