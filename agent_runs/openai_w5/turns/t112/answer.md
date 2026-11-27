The held-out head features do not reliably decode insulin innovations; norepinephrine decoding is weak. That rules out a simple head-only repair. I’ll check the earlier prelude representation, where dose information enters the state. If it retains the signal, a small isolated causal channel may be possible; otherwise, the dose encoder itself will need attention.

```python
print("PRELUDE DATA", E83.shape, innov83.shape, active83.shape,
      "fit/test", int(fit83.sum()), int((~fit83).sum()))
print("FINITE-DOSE DATA")
for k99,v99 in cf83.items():
    print(k99, type(v99).__name__,
          tuple(v99.shape) if hasattr(v99,"shape") else
          {kk99:getattr(vv99,"shape",None) for kk99,vv99 in v99.items()}
          if isinstance(v99,dict) else "")
print("EARLIER FIT REPORT",report83)
for d99 in range(2):
    print("INNOVATION",d99,
          "active std",float(np.std(innov83[active83[:,d99],d99])),
          "inactive rms",float(np.sqrt(np.mean(innov83[~active83[:,d99],d99]**2))))
mu99=E83[fit83].mean(0)
sc99=np.maximum(E83[fit83].std(0),0.05)
ZE99=(E83-mu99)/sc99
models99={}
for alpha99 in [0.1,1.,10.,100.,1000.]:
    model99=Ridge(alpha=alpha99).fit(ZE99[fit83],innov83[fit83])
    pred99=model99.predict(ZE99)
    models99[alpha99]=model99
    stats99=[]
    for d99 in range(2):
        tr99=fit83 & active83[:,d99]
        te99=(~fit83) & active83[:,d99]
        na99=(~fit83) & (~active83[:,d99])
        stats99.append({
            "drug":d99,
            "train_R2":round(r2_score(innov83[tr99,d99],pred99[tr99,d99]),4),
            "test_R2":round(r2_score(innov83[te99,d99],pred99[te99,d99]),4),
            "inactive_rms":round(float(np.sqrt(np.mean(pred99[na99,d99]**2))),4),
            "test_active_sd":round(float(np.std(pred99[te99,d99])),4)})
    print("PRELUDE DECODE",alpha99,stats99)
with open("/work/notes.md","a") as f99:
    f99.write("\n## Runs 97–99: information audit\n")
    f99.write("Head hidden features fail independent insulin innovation decoding; with-code NE decoding reaches only R2=.072. A simple head-only repair is unsupported. Inspecting earlier prelude information before constructing any isolated causal channel. No edits.\n")
print("EDITS",edited())
```
