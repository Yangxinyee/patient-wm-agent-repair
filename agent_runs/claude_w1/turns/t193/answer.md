All three helpers take (x, name). I'll fix q_of and rerun the test. As before, a spread near 0 within each code confirms that q_of reproduces the head query, and then z-draw noise can be added as q_of(eps) − q_of(0).

```python
def q_of(prev, dt, eps=None):
    out_ = prev[:192]; h_ = prev[192:960].reshape(4,192)
    hd_ = h_*np.exp(-dt*np.exp(Wo['core.flow.a']))
    ctx_ = out_ + LNn(hd_.mean(0), 'core.out_norm') + mlp2n(np.array([np.log(dt)/10]), 'gap')
    pr_ = mlp2n(ctx_, 'z_prior'); z_ = pr_[:32] if eps is None else pr_[:32]+np.exp(pr_[32:64])*eps
    return LNn(ctx_ + linn(z_, 'z_head') + Wo['type_emb.weight'][1], 'query_norm'), pr_
Dby = collections.defaultdict(list)
for s_ in list(G3.keys())[:25]:
    df_ = G3[s_]; B_ = bundles_of_stay(df_, RU0); St_ = run_stay(B_); R_ = gen_rows_all(df_, St=St_)
    tb_ = np.array([b_['t'] for b_ in B_])
    for r_ in R_:
        if np.isnan(r_[3]) and r_[2] > -1:
            k_ = int(np.argmin(np.abs(tb_ - r_[2])))
            q_, pr_ = q_of(St_[k_-1], (tb_[k_]-tb_[k_-1])*3600.0)
            Dby[r_[0]].append(r_[8]-q_)
for c_,L_ in Dby.items():
    L_=np.array(L_); print(c_, len(L_), 'spread', np.abs(L_-L_[0]).max().round(5))
print('z prior sd range', np.exp(pr_[32:64]).min().round(3), np.exp(pr_[32:64]).max().round(3))
```