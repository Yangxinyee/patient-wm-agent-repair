The diagnostic stopped while printing a nonnumeric field; it did not change any weights. The fields printed so far show that factual treatment innovations are much smaller than typical doses, which may help protect factual fit. I’ll finish the inspection and test decoding on the independent state sample. Held-out performance, not training fit, will determine whether to proceed.

```python
from sklearn.linear_model import Ridge
for label98,c98 in [("A",corr97A),("B",corr97B)]:
    print("\nCORRECTION", label98)
    for k98,v98 in c98.items():
        if isinstance(v98,(np.ndarray,torch.Tensor)):
            a98=as_np(v98)
            if np.issubdtype(a98.dtype,np.number):
                print(k98,a98.shape,"mean/std",
                      np.round([np.nanmean(a98),np.nanstd(a98)],5).tolist())
            else:
                print(k98,a98.shape,"sample",a98.reshape(-1)[:5].tolist())
        else:
            print(k98,type(v98).__name__,str(v98)[:100])
print("VALUE CENTRES",VOCAB["value_bin_centres"])
print("NORMALISATION",norm_meta)
print("DYNAMIC ESTIMATES",dyn_estimates)

decode98=[]
for feature98 in ["hidden","wc"]:
    XA98=as_np(headTrain0[feature98]).mean(axis=0)[corr97A["rows"]]
    XB98=as_np(headTrainB0[feature98]).mean(axis=0)[corr97B["rows"]]
    ea98=cacheTrain["df"].iloc[corr97A["rows"]]["event"].to_numpy()
    eb98=cacheTrainB["df"].iloc[corr97B["rows"]]["event"].to_numpy()
    for d98,prefix98 in enumerate(["glucose","map"]):
        ma98=np.char.startswith(ea98.astype(str),prefix98)
        mb98=np.char.startswith(eb98.astype(str),prefix98)
        ya98=corr97A["innovations"][ma98,d98]
        yb98=corr97B["innovations"][mb98,d98]
        mu98=XA98[ma98].mean(0)
        sc98=np.maximum(XA98[ma98].std(0),0.05)
        za98=(XA98[ma98]-mu98)/sc98
        zb98=(XB98[mb98]-mu98)/sc98
        for alpha98 in [1.,10.,100.,1000.,10000.]:
            rr98=Ridge(alpha=alpha98).fit(za98,ya98)
            pa98=rr98.predict(za98); pb98=rr98.predict(zb98)
            row98=(feature98,prefix98,alpha98,
                   r2_score(ya98,pa98),r2_score(yb98,pb98),
                   np.std(yb98),np.std(pb98))
            decode98.append(row98)
            print("DECODE",tuple(round(float(x98),4) if isinstance(x98,(float,np.floating)) else x98
                                  for x98 in row98))
print("EDITS",edited())
```
