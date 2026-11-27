The search plateaued at RMSE 3.34 with mean error +0.2. The selected assignment repeats log-age in several columns, which is implausible as a true design. That suggests the remaining error is not a feature-assignment problem but the tool's 4-draw z sampling noise, which no assignment can remove. To check, I'll compare several z treatments against the tool, for both the plausible simple assignment and the searched one:
- my own 4-draw averages, against each other and against a 64-draw expectation;
- the tool's 4-draw average against the 64-draw expectation.

If my 4-draw vs expectation RMSE is ≈ the tool vs expectation RMSE, the head is right up to noise.

```python
def full_eval(cols,stat='none',ty_=1,nz=64,seed=0):
    st=torch.as_tensor({'none':np.zeros(len(fa4)),'has':has_,'nohas':1-has_}[stat].astype(np.int64))
    return head_probs_mc(base1,cod,f8_make(cols),lbt,st,W,ty=ty_,nz=nz,seed=seed)
edges_by={nm:np.array(NUMD[(nm,'value')]['fine_bin_edges']) for nm in MEAS}
obs_bin=np.array([1+np.searchsorted(edges_by[ev],v,side='left') for ev,v in zip(fa4.event.values,fa4.observed.values)])
for cols in [['has','la_s','zl','zero','zero','zero','zero','zero'], ['la_s','la_s','zl','la_s','zero','la_h','la_h','zero']]:
    PE=full_eval(cols,nz=64,seed=1); P4a=full_eval(cols,nz=4,seed=2); P4b=full_eval(cols,nz=4,seed=3)
    pmE=(PE[:,1:]*CENr).sum(1).numpy(); pm4a=(P4a[:,1:]*CENr).sum(1).numpy(); pm4b=(P4b[:,1:]*CENr).sum(1).numpy()
    nllE=-np.log(PE[np.arange(len(fa4)),obs_bin].numpy()); nll4=-np.log(P4a[np.arange(len(fa4)),obs_bin].numpy())
    rm=lambda a,b: np.sqrt(np.mean((a-b)**2))
    print(cols)
    print('  pm: tool-E',round(rm(tool_pm,pmE),3),'4a-E',round(rm(pm4a,pmE),3),'4a-4b',round(rm(pm4a,pm4b),3),'tool-E mean',round(np.mean(tool_pm-pmE),3))
    print('  nll: tool-E',round(rm(fa4.nll.values,nllE),4),'4a-E',round(rm(nll4,nllE),4),'mean tool',round(fa4.nll.mean(),4),'mean E',round(nllE.mean(),4),'mean 4a',round(nll4.mean(),4))
```