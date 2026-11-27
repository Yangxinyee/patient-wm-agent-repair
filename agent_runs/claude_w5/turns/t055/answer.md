Still no match: neither the activation nor the output half explains the error, and the first state row is indeed the 1-atom admission bundle at hour −2.

Next I'll check how far off the prediction is: the correlation between my z_post output and the saved one. I'll also print the norms of the first rows of the other embedding tables (categorical, kq, text), in case the atom vector includes terms I'm leaving out. High correlation would mean a small missing term; low correlation would mean the input layout is wrong.

```python
for n_ in NAMES:
    if n_.startswith('embed.') and n_.endswith('weight') and 'numeric' not in n_:
        w_=Wt[n_]; print(n_, tuple(w_.shape), [round(float(w_[i].norm()),2) for i in range(min(w_.shape[0],6))])
e_=enc(emb_visit(11,3,0,0)[None],'encode')
zp=mlp(torch.cat([gapf(1.0),e_,torch.zeros(192)]),'z_post')[:32]
print('corr', float(np.corrcoef(zp.numpy(),zt.numpy())[0,1]))
print(np.round(zp.numpy()[:8],2)); print(np.round(zt.numpy()[:8],2))
```