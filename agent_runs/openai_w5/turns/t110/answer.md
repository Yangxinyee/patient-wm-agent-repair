The value head exposes 192 hidden features on two independent training-state samples. I’ll check whether those features retain treatment innovations—the dose variation left after accounting for recorded history. Held-out decoding would support a factual-safe correction; failure would suggest that the head lacks the needed dose signal.

```python
corr97A = correction_data(cacheTrain, headTrain0)
corr97B = correction_data(cacheTrainB, headTrainB0)
for label97, c97 in [("A",corr97A),("B",corr97B)]:
    print("\nCORRECTION DATA", label97, type(c97))
    if isinstance(c97,dict):
        for k97,v97 in c97.items():
            if isinstance(v97,(np.ndarray,torch.Tensor)):
                a97 = as_np(v97)
                print(k97, a97.shape,
                      "finite", float(np.isfinite(a97).mean()),
                      "mean/std", np.round([np.nanmean(a97),np.nanstd(a97)],4).tolist()
                      if np.issubdtype(a97.dtype,np.number) else "")
            else:
                print(k97,type(v97).__name__,str(v97)[:200])
    else:
        print(str(c97)[:1000])
print("\nHEAD SHAPES",
      {k97:tuple(v97.shape) for k97,v97 in headTrain0.items()},
      {k97:tuple(v97.shape) for k97,v97 in headTrainB0.items()})
print("VALUE CENTRES", {k97:v97 for k97,v97 in VOCAB["value_bin_centres"].items()})
print("NORM META", norm_meta)
print("DYNAMIC ESTIMATES",dyn_estimates)
print("EDITS",edited())
```
