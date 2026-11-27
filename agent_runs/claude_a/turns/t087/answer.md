The initial-state conventions are settled. A stay starts from h = 0 and out = 0, with dt = 1 s for the first bundle (so log dt = 0), and both the z posterior and the first state then match to float precision. The z-posterior mean also matches on all later bundles (max error 3e-4).

Next I'll write batched versions of the DeepSets encoder and the core step, with all weights taken from a parameter dict so that edited weights can be swapped in. I'll then run a fully self-driven teacher-forced pass over all 100 sel2 stays, from records to atoms to states, and compare every bundle with the tool's states. A max error at float level (≲1e-2 on h of scale 35, ≲1e-4 on out) would confirm the offline model is complete and stable enough for rollouts.

```python
def LNb(x, pre, P): return F.layer_norm(x, (192,), P[pre+'.weight'], P[pre+'.bias'], 1e-5)
def ffnb(x, pre, P): return F.linear(F.gelu(F.linear(x, P[pre+'.ffn.0.weight'], P[pre+'.ffn.0.bias'])), P[pre+'.ffn.2.weight'], P[pre+'.ffn.2.bias'])
def mha_b(xq, xkv, pre, P):
    B_, nq, _ = xq.shape; nk = xkv.shape[1]; Wkv = P[pre+'.kv.weight']
    q = (xq @ P[pre+'.q.weight'].T).view(B_,nq,4,48).transpose(1,2)
    k_ = (xkv @ Wkv[:192].T).view(B_,nk,4,48).transpose(1,2); v_ = (xkv @ Wkv[192:].T).view(B_,nk,4,48).transpose(1,2)
    a = torch.softmax(q @ k_.transpose(-1,-2)/math.sqrt(48), -1)
    return (a @ v_).transpose(1,2).reshape(B_,nq,192) @ P[pre+'.out.weight'].T
def age_emb_P(age_s, P): 
    return F.linear(F.gelu(F.linear(torch.log1p(age_s)[:,None], P['core.age.0.weight'], P['core.age.0.bias'])), P['core.age.2.weight'], P['core.age.2.bias'])
def core_step_b(h, e, u, z, dt, M, Mage, P):
    hf = h * torch.exp(-dt[:,None,None]*torch.exp(P['core.flow.a']))
    zt = F.linear(z, P['z_in.weight'], P['z_in.bias'])
    T = torch.cat([hf + P['core.slots'], e[:,None], u[:,None], zt[:,None]], 1)
    Tn = LNb(T,'core.prelude.0.norm1',P); T = T + mha_b(Tn, Tn, 'core.prelude.0.attn', P)
    T = T + ffnb(LNb(T,'core.prelude.0.norm3',P), 'core.prelude.0', P)
    E = LNb(T[:,:4], 'core.prelude_norm', P); x = hf + E
    Mk = M + age_emb_P(Mage, P)[None]
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        xn = LNb(x, L+'.norm1', P); x = x + mha_b(xn, xn, L+'.attn', P)
        x = x + mha_b(LNb(x, L+'.norm2', P), Mk, L+'.cross', P)
        x = x + ffnb(LNb(x, L+'.norm3', P), L, P)
    sp = F.softplus(P['core.delta_in']); abar = torch.exp(-sp*torch.exp(P['core.a_in']))
    hn = abar*hf + sp*(E @ P['core.b_in.weight'].T) + (x - (hf + E))
    y = hn @ P['core.c_out.weight'].T
    yn = LNb(y,'core.coda.0.norm1',P); y = y + mha_b(yn, yn, 'core.coda.0.attn', P)
    y = y + ffnb(LNb(y,'core.coda.0.norm3',P), 'core.coda.0', P)
    return y, LNb(y.mean(1), 'core.out_norm', P)
def ds_b(av, pre, P):   # av: [B, n, 192] (n>=0)
    B_, n = av.shape[0], av.shape[1]
    s_ = mlpP(av, pre+'.mlp1', P).sum(1)/math.sqrt(n) if n>0 else torch.zeros(B_,192)
    s_ = s_ + F.linear(torch.full((B_,1), math.log1p(n)), P[pre+'.size.weight'], P[pre+'.size.bias'])
    return LNb(mlpP(s_, pre+'.mlp2', P), pre+'.norm', P)
print("empty checks", float((ds_b(torch.zeros(1,0,192),'encode',Po)-e_empty).abs().max()), float((ds_b(torch.zeros(1,0,192),'encode_action',Po)-u_empty).abs().max()))
avs, isact = atom_vectors(AT, {})
isact = torch.as_tensor(np.asarray(isact)).bool(); rowsA = torch.tensor(AT['row'].values)
e_chk = torch.stack([ds_b(avs[(rowsA==r)&~isact][None],'encode',Po)[0] for r in range(200)]); u_chk = torch.stack([ds_b(avs[(rowsA==r)&isact][None],'encode_action',Po)[0] for r in range(200)])
print("ds_b vs eALL/uALL (200 rows)", float((e_chk-eALL[:200]).abs().max()), float((u_chk-uALL[:200]).abs().max()))
# self-driven teacher-forced replica over all sel2 stays
def tf_stay(rows, e_all, u_all, dts_all, mins_all, P):
    h = torch.zeros(1,4,192); o = torch.zeros(1,192); Hs=[]; Os=[]; Zs=[]
    for j, r in enumerate(rows):
        dt = torch.tensor([1.0 if j==0 else float(dts_all[r])])
        cx = ctx_fn(o, h, dt, P); z = zpost_mean(cx, e_all[r][None], u_all[r][None], P)
        mr = rows[max(0,j-127):j+1]
        M = e_all[mr][None]; Mage = torch.tensor((mins_all[r]-mins_all[mr])*60.0, dtype=torch.float32)
        h, o = core_step_b(h, e_all[r][None], u_all[r][None], z, dt, M, Mage, P)
        Hs.append(h[0]); Os.append(o[0]); Zs.append(z[0])
    return torch.stack(Hs), torch.stack(Os), torch.stack(Zs)
t0=time.time(); errH=[]; errO=[]
for s_ in stays_sel2:
    rows = np.where(SB['stay_id'].values==s_)[0]
    Hs_, Os_, Zs_ = tf_stay(rows, eALL, uALL, SB['dts'].values, SB['minute'].values.astype(np.float64), Po)
    errH.append(float((Hs_-HS[rows]).abs().max())); errO.append(float((Os_-OUT[rows]).abs().max()))
print("self-driven TF: max |dH|", max(errH), "max |dOut|", max(errO), "time", round(time.time()-t0,1))
```