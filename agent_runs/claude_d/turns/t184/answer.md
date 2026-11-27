The tensors I need are there: z_head (32 to 192), query_norm, type_emb and last_numeric. I rebuild with_code as query_norm(context + z_head(z) + type_emb[1]) + register terms. Then I check whether the stored WC0 is reproduced with z set to the prior mean. If the difference is near zero, I can simulate the tool's four prior draws of z offline and measure how much decoder noise the sampling adds.

```python
zhW = load_original('z_head.weight').numpy().astype(np.float64); zhb = load_original('z_head.bias').numpy().astype(np.float64)
qnw = load_original('query_norm.weight').numpy().astype(np.float64); qnb = load_original('query_norm.bias').numpy().astype(np.float64)
temb = load_original('type_emb.weight').numpy().astype(np.float64)
def LN(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x-m)**2).mean(-1, keepdims=True)
    return (x-m)/np.sqrt(v+eps)*w + b
def wc_from(z, ctx=HI['ctx'], reg=HI['reg']):
    view = ctx + z@zhW.T + zhb
    return LN(view + temb[1], qnw, qnb) + reg
W_mu = wc_from(HI['zmu'])
print('diff vs WC0 (z=mu)', float(np.abs(W_mu-WC0).max()), 'zhW', zhW.shape, 'temb', temb.shape)
rng = np.random.default_rng(0)
zs = HI['zmu'] + np.exp(HI['zls'])*rng.standard_normal(HI['zmu'].shape)
W_s = wc_from(zs)
print('wc rms change one draw', round(float(np.sqrt(((W_s-WC0)**2).mean())),3), 'wc rms', round(float(np.sqrt((WC0**2).mean())),3))
```