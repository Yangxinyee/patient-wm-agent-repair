The unit search confirms mg/dL = row 3 and mmHg = row 2, with the strict "<" bin rule, but the RMSE stays at 0.08. The remaining error is not in the unit mapping, so next I will test the structure of the set encoder itself:
- where its LayerNorm sits;
- where the size term enters;
- whether pooling is a sum divided by √n, a plain mean, or a plain sum.

If one variant drops the error well below 0.08, the encoder form is identified.

```python
CFG['unit'] = {'glucose_finger': 3, 'glucose_lab': 3, 'map_arterial': 2, 'map_cuff': 2}
ENC_V = 'v1'
def set_enc(embs, p):
    if len(embs) == 0: return torch.zeros(192, dtype=torch.float64)
    X_ = torch.stack(embs); n_ = X_.shape[0]
    h1 = mlpm(X_, p+'.mlp1','enc'); sz = lin(torch.tensor([np.log1p(n_)], dtype=torch.float64), p+'.size')
    if ENC_V == 'v1': return LN(mlpm(h1.sum(0)/np.sqrt(n_) + sz, p+'.mlp2','enc'), p+'.norm')
    if ENC_V == 'v2': return mlpm(LN(h1.sum(0)/np.sqrt(n_) + sz, p+'.norm'), p+'.mlp2','enc')
    if ENC_V == 'v3': return LN(mlpm(h1.sum(0)/np.sqrt(n_), p+'.mlp2','enc') + sz, p+'.norm')
    if ENC_V == 'v4': return LN(mlpm(h1.mean(0) + sz, p+'.mlp2','enc'), p+'.norm')
    if ENC_V == 'v5': return LN(mlpm(h1.sum(0) + sz, p+'.mlp2','enc'), p+'.norm')
    if ENC_V == 'v6': return LN(mlpm(h1.sum(0)/np.sqrt(n_) + sz, p+'.mlp2','enc') + h1.sum(0)/np.sqrt(n_) + sz, p+'.norm')
for ENC_V in ['v1','v2','v3','v4','v5','v6']:
    print(ENC_V, zerr(ctx_nf))
ENC_V = 'v1'
```