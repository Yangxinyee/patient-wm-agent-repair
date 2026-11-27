Those extreme factual errors indicate a scaling or encoding mismatch, not a usable dose estimate. No weights have changed. I’ll compare the augmented features directly with cached action vectors and reproduce the earlier fit’s predictions. Exact vector matches and prediction agreement are prerequisites for continuing.

```python
from sklearn.neighbors import NearestNeighbors

r48_X = np.asarray(r41_X,dtype=np.float32)
r48_scale = np.asarray(r41_uscale,dtype=np.float32)
r48_U = np.asarray(u39_zeroed,dtype=np.float32)
print("X row-norm quantiles:",np.quantile(np.linalg.norm(r48_X,axis=1),[0,.1,.5,.9,1]))
print("actual U row-norm quantiles:",np.quantile(np.linalg.norm(r48_U,axis=1),[0,.1,.5,.9,1]))
print("scale quantiles:",np.quantile(r48_scale,[0,.01,.1,.5,.9,1]))
print("X coordinate-std quantiles:",np.quantile(r48_X.std(0),[0,.1,.5,.9,1]))
print("actual U coordinate-std quantiles:",np.quantile(r48_U.std(0),[0,.1,.5,.9,1]))

r48_check = np.linspace(0,len(r48_X)-1,1500,dtype=int)
print("\n=== REPRODUCING EARLIER PREDICTIONS ===")
for label,xx in [("X",r48_X[r48_check]),
                 ("X / scale",r48_X[r48_check]/r48_scale),
                 ("X * scale",r48_X[r48_check]*r48_scale)]:
    pred = r41_action_fit.predict(xx)
    print(label,"RMSE versus stored predictions:",
          float(np.sqrt(np.mean((pred-np.asarray(r41_pred)[r48_check])**2))))

r48_query_idx = np.flatnonzero(np.any(r48_U != 0,axis=1))[::31]
r48_query = r48_U[r48_query_idx]
r48_matches = {}
print("\n=== NEAREST PHYSICAL ACTION-VECTOR MATCHES ===")
for label,xx in [("X",r48_X),("X * scale",r48_X*r48_scale),
                 ("X / scale",r48_X/r48_scale)]:
    nn = NearestNeighbors(n_neighbors=1,algorithm="brute",n_jobs=4).fit(xx)
    dist,ind = nn.kneighbors(r48_query)
    r48_matches[label]=(dist[:,0],ind[:,0])
    print(label,"distance quantiles:",np.quantile(dist[:,0],[0,.1,.5,.9,1]),
          "fraction below .001:",float(np.mean(dist[:,0]<.001)))
print("\n=== EARLIER LINEAR EXTRACTOR ON ACTUAL U ===")
r48_prior_pred = r41_action_fit.predict(r48_U/r48_scale)
for j,drug in enumerate(["insulin","norepinephrine"]):
    active=np.asarray(r41_present)[:,j]>.5
    print(drug,"active rate RMSE:",
          float(np.sqrt(np.mean((r48_prior_pred[active,j]-r41_rates[active,j])**2))),
          "presence quantiles active/inactive:",
          np.quantile(r48_prior_pred[active,j+2],[.01,.5,.99]),
          np.quantile(r48_prior_pred[~active,j+2],[.01,.5,.99]))
```
