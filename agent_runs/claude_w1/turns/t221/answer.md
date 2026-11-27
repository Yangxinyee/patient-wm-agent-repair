With both own-drug edits constrained, the decoders have noise per unit response of 1.33 (insulin) and 0.26 (NE). At the needed gains (about -2 glucose units per insulin unit and about +3.4 MAP units per NE unit), the plain Gaussian estimate of factual cost is about 0.005 + 0.005 nats. That is roughly twice the budget. However, part of the decoded variation is real dose signal, which should lower the cost, and gating each correction to the one readout code (finger, arterial) cuts it further.

The exact cost should come from replica head probabilities on setA's teacher-forced rows. I need to find which stored arrays hold those rows' head inputs, codes, last bins and observed bins.

```python
c5=[k for k,v in globals().items() if isinstance(v,np.ndarray) and v.shape[:1]==(5177,) and not k.startswith('_')]
for k in c5:
    v=globals()[k]; vv=v.astype(float)
    print(k, v.shape, v.dtype, 'min',round(np.nanmin(vv),2),'max',round(np.nanmax(vv),2),'mean',round(np.nanmean(vv),3))
print('Xf norm',round(np.linalg.norm(Xf[:2000],axis=1).mean(),2),'wc_ norm',round(np.linalg.norm(wc_,axis=1).mean(),2),'TR',TR.shape,'W1',W1.shape,'W2',W2.shape)
```