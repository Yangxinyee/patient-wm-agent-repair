Dropping the time flow from the context improved things clearly: posterior correlation 0.85, error 0.045. This held under both gap inputs that work well, log1p of dt in hours and log of dt in seconds over 10.

Taken literally, the decay of h should have been applied. Two explanations fit:
- the context uses h before the decay;
- the decay uses a time unit that makes it negligible over one hour.

Most of these bundles have the same gap (55 minutes after the action bundle). So the gap network effectively sees a single input value. Rather than guess its formula, I will scan that value directly, together with a multiplier on dt inside the time flow.

If the best combination brings the error close to zero, the context is solved. A residual error that does not shrink would mean another missing ingredient.

```python
dts = np.array([c_[2] for c_ in cand2]); print('dt values (s):', np.unique(np.round(dts)), np.bincount(np.searchsorted(np.unique(np.round(dts)), np.round(dts))))
sel_ = np.abs(dts - 3300) < 1
def post_err_g(gval, fmul):
    P_ = []
    for (idx_, ip, dt, g_), e_k in zip([c_ for c_, s__ in zip(cand2, sel_) if s__], [e_ for e_, s__ in zip(E_list, sel_) if s__]):
        x_ = X[ip].astype(np.float64); out = x_[:192]; h = x_[192:960].reshape(4, 192)
        hf = h * np.exp(-dt * fmul * flow_k)
        c_ = out + ln(hf.mean(0), W['core.out_norm.weight'], W['core.out_norm.bias']) + mlp(np.array([gval]), 'gap')
        P_.append(mlp(np.concatenate([c_, e_k, np.zeros(192)]), 'z_post')[:32])
    P_ = np.array(P_); T_ = T_all[sel_]
    return np.abs(P_ - T_).mean()
res_ = []
for fmul in [0, 1 / 3600, 1 / 60, 0.1, 1]:
    for gval in np.linspace(-1.0, 3.0, 21):
        res_.append((post_err_g(gval, fmul), round(gval, 2), fmul))
res_.sort(); print(res_[:8])
```