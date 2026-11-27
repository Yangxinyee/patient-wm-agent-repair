The no-action u_k (norm 11.6) is on the same scale as the drug u_k vectors (11.7), so it is a valid LayerNorm output. I built approximate e_k for 520 pure-measurement bundles.

Now I'll run the prelude replica on those bundles and print the slots' attention per head on e_k, u_k (constant), z and the slots. This decides how to gate an injection:
- If heads 1–3 still look at u_k, a u_k-value carrier fires only where u_k differs from the constant, which is exactly the drug bundles.
- If they look at e_k, that path also carries reading content, and the gate must be robust to it.

```python
keep = np.array(keep)
pk = keep - 1
hp = torch.tensor(X3[pk][:, 192:960]).reshape(-1,4,192)
dtk = torch.tensor((hr[keep]-hr[pk])*3600.0, dtype=torch.float32)[:,None,None]
htf_n = hp*torch.exp(-dtk*torch.exp(fa))
ek_n = torch.stack([e.float() for e in E_nd])
zk_n = torch.tensor(Z3[keep]).float()
zt_n = zk_n @ Wd['z_in.weight'].T + Wd['z_in.bias']
TOKn = torch.cat([htf_n + Wd['core.slots'][None], ek_n[:,None], u_const.expand(len(keep),192)[:,None], zt_n[:,None]], 1)
En, attn_n, vhn, xn_ = prelude(TOKn, ret=True)
for hh in range(4):
    print(hh, 'e,u,z:', np.round(attn_n[:,hh,:4,4:].mean((0,1)).numpy(),3), 'slots', np.round(attn_n[:,hh,:4,:4].sum(-1).mean().item(),3),
          'min u', round(attn_n[:,hh,:4,5].min().item(),3))
```