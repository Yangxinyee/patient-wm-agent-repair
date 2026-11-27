The step failed on a tensor name: the latent projections are top-level tensors (`z_in.*`, `z_head.*`, `z_post.*`, `z_prior.*`), not under `core.`. I'll fix the name and rerun the same variant search. The confirm/refute criterion is unchanged: one variant should reproduce `HS`/`OUT` to float precision.

```python
print([k for k in NAMES if k.startswith('z_')])
def core_step(h, e, u, z, dt, M, Mage, var):
    hf = h * torch.exp(-dt*torch.exp(Po['core.flow.a']))
    zt = F.linear(z, Po['z_in.weight'], Po['z_in.bias'])
    T = torch.cat([hf + Po['core.slots'], e[None], u[None], zt[None]], 0)
    Tn = LNp(T,'core.prelude.0.norm1'); T = T + mha(Tn, Tn, 'core.prelude.0.attn')
    T = T + ffn(LNp(T,'core.prelude.0.norm3'), 'core.prelude.0')
    E = LNp(T[:4], 'core.prelude_norm'); x = hf + E
    ae = age_emb(Mage, var['age'])
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        xn = LNp(x, L+'.norm1'); x = x + mha(xn, xn, L+'.attn')
        xn = LNp(x, L+'.norm2'); km = var['km']
        if km=='kadd': x = x + mha(xn, M, L+'.cross', kadd=ae)
        elif km=='kin': x = x + mha(xn, M, L+'.cross', kin=ae)
        else: x = x + mha(xn, M, L+'.cross', kin=ae, vin=ae)
        x = x + ffn(LNp(x, L+'.norm3'), L)
    sp = F.softplus(Po['core.delta_in']); abar = torch.exp(-sp*torch.exp(Po['core.a_in']))
    hn = abar*hf + sp*(E @ Po['core.b_in.weight'].T) + (x - (hf + E))
    y = hn @ Po['core.c_out.weight'].T
    yn = LNp(y,'core.coda.0.norm1'); y = y + mha(yn, yn, 'core.coda.0.attn')
    y = y + ffn(LNp(y,'core.coda.0.norm3'), 'core.coda.0')
    return y, LNp(y.mean(0), 'core.out_norm')
res_v = []
for km, ag, mem in itertools.product(['kadd','kin','kvin'], ['raw','div10'], ['prev','incl']):
    var = {'km':km,'age':ag}; ey=[]; eo=[]
    for r in test_rows:
        pr_i = int(SB['prev'].values[r])
        same = np.where(sids_==sids_[r])[0]; mem_rows = same[same<r] if mem=='prev' else same[same<=r]; mem_rows = mem_rows[-128:]
        M = eALL[mem_rows]; Mage = torch.tensor((mins_[r]-mins_[mem_rows])*60.0)
        y, o = core_step(HS[pr_i], eALL[r], uALL[r], ZP[r], float(SB['dts'].values[r]), M, Mage, var)
        ey.append(float((y-HS[r]).abs().mean())); eo.append(float((o-OUT[r]).abs().mean()))
    res_v.append((km,ag,mem,round(np.mean(ey),6),round(np.max(ey),6),round(np.mean(eo),6)))
for rv in sorted(res_v, key=lambda t:t[3]): print(rv)
print("scale |HS|", float(HS.abs().mean()), "|OUT|", float(OUT.abs().mean()))
```