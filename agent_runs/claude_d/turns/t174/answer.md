The offline head reproduces the stored means and NLLs exactly (difference 0). The mean is over classes 1 to 16, and class 0 has negligible mass. So I can now predict exactly how a value-head weight edit changes every factual prediction for these 400 stays.

Next I take stock of the inputs for the edit: which record tables are in the kernel (needed to build the norepinephrine surprise), and how the insulin-surprise columns look. I also compute, for every reading, the gradient of the predicted mean with respect to with_code. That gradient lets me pick an injection direction that moves glucose predictions but not MAP predictions.

```python
dfs = {k:v.shape for k,v in globals().items() if isinstance(v, pd.DataFrame) and not k.startswith('_')}
print(dfs)
e1 = RD2['eps_ins_1']
print('eps1 nan', int(e1.isna().sum()), 'zero', int((e1==0).sum()), 'sd nz', round(float(e1[e1!=0].std()),3), 'n nz', int((e1.fillna(0)!=0).sum()))
A = WC0@W1.T + b1
gp = 0.5*(1+erf(A/np.sqrt(2))) + A*np.exp(-0.5*A**2)/np.sqrt(2*np.pi)   # gelu'
m0 = mA
G = (P[:,1:]*(cen - m0[:,None])) @ W2[1:]      # dmean/dhidden (n,192)
JAC = (G*gp) @ W1                                # dmean/dwc (n,192)
evc = RD2['event'].astype(str).values
isg = np.isin(evc, ['glucose_finger','glucose_lab']); ism = ~isg
print('J norms g/m', round(float(np.linalg.norm(JAC[isg],axis=1).mean()),2), round(float(np.linalg.norm(JAC[ism],axis=1).mean()),2))
```