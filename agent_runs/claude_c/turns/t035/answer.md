I have the states: 4,356 bundles from 100 stays. Each row holds out (192), the four h slots (768) and the z posterior mean (32), with the stay, hour, drug flags and atom count.

Next I will load them and map rows to bundles (reading bundles at whole hours, drug bundles at h + 0.083). Then I will start the z analysis. For insulin bundles I will regress the z posterior mean on the within-tier dose (controlling for the reading and tier). I will also test whether z depends on the tier base at a fixed dose, which would indicate a deviation code. A clear dose-dependent direction in z (high R² from dose) would be a candidate channel to amplify.

```python
S = np.load('/work/fwd_3/states_original.npz')
print({k: S[k].shape for k in S.files})
X = S['x']; sidv = S['stay_id']; hrv = S['hour']; hi = S['has_insulin']; hn = S['has_norepinephrine']; na = S['n_atoms']
st = pd.DataFrame({'stay_id':sidv, 'hour':hrv, 'ins':hi, 'ne':hn, 'na':na})
print(st.head(15).to_string())
print(st.hour.apply(lambda x: round(x - np.floor(x),4)).value_counts().head())
print(st.groupby([st.ins, st.ne]).size())
# norms
print("out norm", np.linalg.norm(X[:,:192],axis=1).mean(), " slot norms", [np.linalg.norm(X[:,192+192*s:384+192*s],axis=1).mean() for s in range(4)], " z norm", np.linalg.norm(X[:,960:],axis=1).mean())
```