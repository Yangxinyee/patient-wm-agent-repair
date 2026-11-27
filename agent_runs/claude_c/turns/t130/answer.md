Pass 3 has the original model's states for 100 tuning stays. I'll load them and check their structure, so I can look for drug-selective directions in h and out using exogenous dose noise.

```python
S=np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
for k in S.files: print(k, S[k].shape, S[k].dtype)
```