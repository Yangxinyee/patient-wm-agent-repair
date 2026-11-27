On clean data the linear readout still distorts factual predictions badly:
- At factual weight 1: about 18 mg/dL RMS, costing 0.074 nats per reading overall.
- At factual weight 20: 5.5 mg/dL and 0.020 nats, still four times the 0.005 budget, while counterfactual recovery falls to about half.

So a linear readout of `with_code` can't yet separate counterfactual from factual histories. Before choosing a remedy I need to know where the factual distortion sits:
- If it is concentrated in early hours or in stays and hours with no drug exposure (states the counterfactual rows never cover), adding coverage or gating on exposure would help.
- If it is spread over exposed hours, the representation itself doesn't carry the dose-deviation signal linearly, and I need a different mechanism.

```python
ix,oof,mn=fit_eval(isG,vG,100,wf=5)
f=NONE[ix]; d=mn-M.mean_f.values[ix]; h=M.hour.values[ix]; st=M.stay.values[ix]
expo={s:any(v>0 for j,v in drug_hist(s,'insulin').items()) for s in BUN}
first={s:min([j for j,v in drug_hist(s,'insulin').items() if v>0],default=99) for s in BUN}
ex=np.array([h_>first[s_] for h_,s_ in zip(h,st)])
for nm,g in (('h<=6',f&(h<=6)),('6-12',f&(h>6)&(h<=12)),('>12',f&(h>12)),('exposed',f&ex),('not exposed',f&~ex)):
    print(nm,'n',g.sum(),'mean %.2f rms %.2f'%(d[g].mean(),np.sqrt((d[g]**2).mean())))
```