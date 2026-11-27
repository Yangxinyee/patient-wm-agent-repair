Run 86 gave R² of about 0 at every lag. That run misaligned the lags, though: for a reading bundle at hour t, "lag 0" pointed to the dose at t:05, which comes after the reading. Ridge with α=10 on 992 features may also be overfitting a weak signal. I'll redo the test with each bundle aligned to the most recent dose at or before it and several regularization strengths. If lag-0/1 R² comes back to around 0.3 and older lags stay near 0, the state carries only a short-lived trace of the actual dose. That would mean a head-only edit cannot produce the hours-long effect, and I would need to edit the core.

```python
t0=np.floor(hr3-5/60+1e-9).astype(int)
Y=np.zeros((len(X),L))
for i,(s,t) in enumerate(zip(sid3,t0)):
    e=EPS[s]
    for j in range(L):
        if 0<=t-j<len(e): Y[i,j]=e[t-j]
for alpha in [30,300,3000]:
    for nm,sl in [('h',slice(192,960)),('all',slice(0,992))]:
        r2=[]
        for j in range(L):
            pr=np.zeros(len(X))
            for a,b in kf.split(X):
                pr[b]=Ridge(alpha=alpha).fit(X[a,sl],Y[a,j]).predict(X[b,sl])
            r2.append(1-np.mean((Y[:,j]-pr)**2)/np.var(Y[:,j]))
        print(alpha,nm,np.round(r2,2))
```