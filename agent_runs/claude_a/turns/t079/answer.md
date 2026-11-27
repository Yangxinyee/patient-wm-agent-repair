`make_atoms` reproduces the atom and bundle tables exactly: atom counts, `e`/`u` encodings, dt and minutes all match. I can now encode any stay, including the tuning stays and edited-dose versions.

Next I'll write a batched value head: ctx from the previous state → z from the prior → view → query → with_code (register features) → value-bin MLP plus transition table → bin probabilities and predicted mean. I'll validate it against the tool's factual predicted means in `R` (5,802 readings). The tool averages over 4 random z draws, so my 64-draw average should agree up to that sampling noise (rmse of a few tenths of a unit at most). A systematic gap would point to a missing term.

```python
def zsc_v(code, x): return np.array([zsc(code,'value',float(v)) if np.isfinite(v) else 0.0 for v in x], dtype=np.float32)
def fbin_v(code, x): return np.array([fbin(code,'value',float(v)) if np.isfinite(v) else 0 for v in x], dtype=np.int64)
print("zsc test", zsc('glucose_finger','value',150.0), fbin('glucose_finger','value',150.0), centM.shape)
def mlp2(x, pre): return F.linear(F.gelu(F.linear(x, P_[pre+'.0.weight'], P_[pre+'.0.bias'])), P_[pre+'.2.weight'], P_[pre+'.2.bias'])
P_ = Po
def head_value(out_prev, h_prev, dt, code_idx, has_last, age_s, z_last, last_vbin, eps, P=None):
    global P_
    P_ = Po if P is None else P
    hf = h_prev * torch.exp(-dt[:,None,None]*torch.exp(P_['core.flow.a']))
    ctx = out_prev + F.layer_norm(hf.mean(1),(192,),P_['core.out_norm.weight'],P_['core.out_norm.bias'],1e-5) + mlp2(torch.log(dt)[:,None]/10, 'gap')
    pr = mlp2(ctx, 'z_prior'); mu, lsd = pr[:,:32], pr[:,32:]
    z = mu[:,None,:] + torch.exp(lsd)[:,None,:]*eps            # B,nz,32
    view = ctx[:,None,:] + F.linear(z, P_['z_head.weight'], P_['z_head.bias'])
    q = F.layer_norm(view + P_['type_emb.weight'][1], (192,), P_['query_norm.weight'], P_['query_norm.bias'], 1e-5)
    f8 = torch.zeros(len(dt), 8); f8[:,0] = has_last; f8[:,3] = torch.log1p(age_s)/10*has_last; f8[:,6] = z_last
    wc = q + (P_['embed.code.weight'][code_idx] + F.linear(f8, P_['last_numeric.weight'], P_['last_numeric.bias']) + P_['embed.bins.value.weight'][last_vbin])[:,None,:]
    lg = mlp2(wc, 'attr_heads.value_bin') + P_['embed_transition.value_bin.weight'][code_idx*17 + last_vbin][:,None,:]
    p = torch.softmax(lg, -1).mean(1)
    mean = (p[:,1:] * centM[code_idx]).sum(1) / p[:,1:].sum(1).clamp_min(1e-9) * 0 + (p[:,1:] * centM[code_idx]).sum(1)
    return p, mean
Rr = R.copy()
cidx = torch.tensor(Rr['event'].map(cid).values)
hasl = torch.tensor(Rr['lastv'].notna().values.astype(np.float32))
ages = torch.tensor(np.nan_to_num((Rr['hour'].values - Rr['lasth'].values)*3600.0).astype(np.float32))
zl = torch.tensor(np.concatenate([zsc_v(ev, Rr.loc[Rr.event==ev,'lastv'].values) for ev in []]) if False else np.zeros(len(Rr),np.float32))
lvb = np.zeros(len(Rr), np.int64); zl_np = np.zeros(len(Rr), np.float32)
for ev in Rr['event'].unique():
    m_ = (Rr['event']==ev).values
    zl_np[m_] = zsc_v(ev, Rr.loc[m_,'lastv'].values); lvb[m_] = fbin_v(ev, Rr.loc[m_,'lastv'].values)
zl = torch.tensor(zl_np); lvbt = torch.tensor(lvb)
rp = torch.tensor(Rr['row_prev'].values.astype(np.int64)); dtt = torch.tensor(Rr['dt'].values.astype(np.float32))
torch.manual_seed(0); eps64 = torch.randn(len(Rr), 64, 32)
p_h, m_h = head_value(OUT[rp], HS[rp], dtt, cidx, hasl, ages, zl, lvbt, eps64)
dm = m_h.numpy() - Rr['predicted_mean'].values
for ev in Rr['event'].unique():
    m_ = (Rr['event']==ev).values; print(ev, m_.sum(), "rmse", round(float(np.sqrt(np.mean(dm[m_]**2))),3), "mean diff", round(float(dm[m_].mean()),3))
obs_b = np.zeros(len(Rr), np.int64)
for ev in Rr['event'].unique():
    m_ = (Rr['event']==ev).values; obs_b[m_] = fbin_v(ev, Rr.loc[m_,'observed'].values)
nll_h = -torch.log(p_h[torch.arange(len(Rr)), torch.tensor(obs_b)]).numpy()
print("NLL mine", nll_h.mean(), "tool", Rr['nll'].mean(), "corr", np.corrcoef(nll_h, Rr['nll'].values)[0,1])
```