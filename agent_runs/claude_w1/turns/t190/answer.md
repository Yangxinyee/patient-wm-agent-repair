Rows are (code, stay, hour, previous value, current value, 0, 0, 0, with_code). States are (54, 992) per stay: out + 4 slots of h + z posterior mean.

To add z-draw noise I need, per row, the state after the preceding bundle and the gap dt, and a q_of(prev, dt, eps) that reproduces wc_vec's q. Test: rows whose code has no earlier reading carry no last-value information. So with_code − q_of(eps=0) should be identical across different stays for the same code. If the maximum difference is about 0, q_of is exact and I can simulate prior draws.

```python
print(list(B0_[0].keys()))
tkey = [k_ for k_ in B0_[0].keys() if 'hour' in k_ or k_ in ('t','tm','tmin','time')]
print('tkey', tkey, [B0_[i][tkey[0]] for i in range(4)] if tkey else '')
def q_of(prev, dt, eps=None):
    out_ = prev[:192]; h_ = prev[192:960].reshape(4,192)
    hd_ = h_*np.exp(-dt*np.exp(Wo['core.flow.a']))
    ctx_ = out_ + LNn('core.out_norm', hd_.mean(0)) + mlp2n('gap', np.array([np.log(dt)/10]))
    pr_ = mlp2n('z_prior', ctx_); z_ = pr_[:32] if eps is None else pr_[:32]+np.exp(pr_[32:64])*eps
    return LNn('query_norm', ctx_ + linn('z_head', z_) + Wo['type_emb.weight'][1]), pr_
print(Wo['core.flow.a'].shape)
```