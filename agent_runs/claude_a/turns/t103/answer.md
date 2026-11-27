The pieces of the teacher-forced pipeline check out:

- `ctx_fn` plus `zpost_mean` reproduce the stored z posterior (error 3e-4).
- `head_value` reproduces the stored 4-sample bin probabilities exactly. The resulting NLL is 1.40449, against 1.40394 from the tool, which uses different z draws.

So I can now run a fully sequential teacher-forced pass with any edited weights.

**This run.**
1. Write the sequential state pass, `tf_states`, batched across stays.
2. Check it against the stored states and the baseline NLL.
3. Measure what repurposing costs. I will kill a group of the 8 least-read h dimensions, so they are held at 0 and nothing flows through them, and see how much the factual NLL worsens.

**How I will read it.**
- *Pipeline confirmed:* the original weights reproduce HS (error around 0.01) and an NLL of about 1.4045.
- *Plan viable:* killing the 8 dimensions costs well under the 0.005 tolerance (ideally below 0.001), which leaves room for the memory channels.
- *Plan in trouble:* a cost near or above 0.005. Then I would need fewer dimensions, or a different choice of dimensions.

```python
import copy
print('codes', cid)
stay_rows = [np.where(SB['stay_id'].values == s_)[0] for s_ in stays_SB]
def tf_states(P, stay_rows=stay_rows, e_all=eALL, u_all=uALL):
    N = len(SB); Hn = torch.zeros(N, 4, 192); On = torch.zeros(N, 192); Zn = torch.zeros(N, 32)
    L = max(len(r) for r in stay_rows)
    with torch.no_grad():
        for k in range(L):
            idx = [r for r in stay_rows if len(r) > k]
            rows_k = np.array([r[k] for r in idx])
            if k == 0:
                hprev = torch.zeros(len(rows_k), 4, 192); oprev = torch.zeros(len(rows_k), 192); dt = torch.ones(len(rows_k))
            else:
                pr = np.array([r[k-1] for r in idx]); hprev = Hn[pr]; oprev = On[pr]
                dt = torch.tensor(SB['dts'].values[rows_k], dtype=torch.float32)
            ctx = ctx_fn(oprev, hprev, dt, P); z = zpost_mean(ctx, e_all[rows_k], u_all[rows_k], P)
            mem = torch.tensor(np.stack([r[:k+1] for r in idx]))
            t = tsec[mem]; ages = (t[:, -1:] - t).clamp(min=0)
            y, o = step_b(hprev, e_all[rows_k], u_all[rows_k], z, dt, e_all[mem], ages, torch.ones(len(rows_k), k+1, dtype=torch.bool), P)
            Hn[rows_k] = y; On[rows_k] = o; Zn[rows_k] = z
    return Hn, On, Zn
obs_t = torch.tensor(obs_b)
def tf_nll(P, ret=False):
    Hn, On, Zn = tf_states(P)
    ph, mh = head_value(On[prevrow], Hn[prevrow], dtv, cidx, hasl, ages, zl, lvbt, eps4, P)
    nll = -torch.log(ph[torch.arange(len(obs_t)), obs_t])
    return (nll, ph, mh, Hn, On) if ret else nll
t0_ = time.time()
nll0, ph0, mh0, H0, O0 = tf_nll(PO, ret=True)
print('replica vs stored H max', (H0 - HS).abs().max().item(), ' NLL', nll0.mean().item(), ' time', round(time.time() - t0_, 1))
codes_np = cidx.numpy() if torch.is_tensor(cidx) else np.asarray(cidx)
for c_ in np.unique(codes_np): print('code', c_, 'n', (codes_np == c_).sum(), 'nll', nll0[codes_np == c_].mean().item())
def kill_dims(P, D):
    Q_ = {k: v.clone() for k, v in P.items()}
    D = torch.tensor(D)
    Wc = Q_['core.c_out.weight']; Wc[D, :] = 0; Wc[:, D] = 0
    Q_['core.coda.0.attn.out.weight'][D, :] = 0; Q_['core.coda.0.ffn.2.weight'][D, :] = 0; Q_['core.coda.0.ffn.2.bias'][D] = 0
    return Q_
for D in ([1, 33, 17, 12, 4, 132, 36, 66], [1, 33, 17, 12], [101, 2, 137, 84, 16, 9, 36, 66]):
    t0_ = time.time()
    nllk = tf_nll(kill_dims(PO, D))
    dd = (nllk - nll0)
    print('kill', D, 'dNLL all %.5f' % dd.mean().item(), ' per code', {int(c_): round(dd[codes_np == c_].mean().item(), 5) for c_ in np.unique(codes_np)}, round(time.time() - t0_, 1), 's')
```