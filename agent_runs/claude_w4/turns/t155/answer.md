99 of the 100 admission bundles are now encoded. 100 rows still fail, likely the hospital_admission bundles at hour −2 that my lookup misses. I'll give those a visit atom directly, so the memory M is complete for every stay.

Then I'll run the full replica (prelude → two recurrent blocks with cross-attention to M → gated update → c_out → coda) at all 2,319 drug bundles. Eight memory variants:
- current e included or not;
- age in seconds or hours;
- age added after or before the key projection.

I'll print each variant's relative error of h and out against the GPU states. A variant with error well below the rest identifies the structure. The size of that error tells me how much the approximate measurement e vectors cost.

```python
for i in np.where(src == 'fail')[0]:
    e_all[i] = enc_e([visit_atom(11)]).reshape(-1).float(); src[i] = 'adm'
LNw = lambda x, p: LNt(x, Wc[p+'.weight'], Wc[p+'.bias'])
def mha_m(xq, xkv, Wq, Wkv, Wo, kadd=None, mask=None, kpre=None):
    N, Tq, _ = xq.shape; Tk = xkv.shape[1]
    q = xq @ Wq.T
    k = (xkv + (kpre if kpre is not None else 0)) @ Wkv[:192].T; v = xkv @ Wkv[192:].T
    if kadd is not None: k = k + kadd
    qh = q.reshape(N,Tq,4,48).transpose(1,2); kh = k.reshape(N,Tk,4,48).transpose(1,2); vh = v.reshape(N,Tk,4,48).transpose(1,2)
    s = qh @ kh.transpose(-1,-2)/math.sqrt(48)
    if mask is not None: s = s.masked_fill(~mask[:,None,None,:], -1e9)
    return (torch.softmax(s, -1) @ vh).transpose(1,2).reshape(N,Tq,192) @ Wo.T
L = 64
def build_mem(rws, incl, unit):
    Mm = torch.zeros(len(rws), L, 192); Ag = torch.zeros(len(rws), L); Mk = torch.zeros(len(rws), L, dtype=torch.bool)
    for n, i in enumerate(rws):
        j0 = i
        while j0 - 1 >= 0 and st[j0-1] == st[i]: j0 -= 1
        idx = np.arange(j0, i + (1 if incl else 0))[-128:]
        Mm[n, :len(idx)] = e_all[idx]; Ag[n, :len(idx)] = torch.tensor((hr[i] - hr[idx]) * unit, dtype=torch.float32); Mk[n, :len(idx)] = True
    return Mm, Ag, Mk
def step_full(rws, E, htf, incl=True, unit=3600.0, pre=False, dE=None):
    Mm, Ag, Mk = build_mem(rws, incl, unit)
    ae = age_emb(Ag)
    x = htf + E
    for p in ['core.blocks.0.0.', 'core.blocks.0.1.']:
        n1 = LNw(x, p+'norm1'); x = x + mha_m(n1, n1, Wc[p+'attn.q.weight'], Wc[p+'attn.kv.weight'], Wc[p+'attn.out.weight'])
        n2 = LNw(x, p+'norm2')
        x = x + mha_m(n2, Mm, Wc[p+'cross.q.weight'], Wc[p+'cross.kv.weight'], Wc[p+'cross.out.weight'],
                      kadd=None if pre else ae, mask=Mk, kpre=ae if pre else None)
        x = x + ffn(LNw(x, p+'norm3'), p)
    sp = torch.nn.functional.softplus(Wc['core.delta_in']); abar = torch.exp(-sp*torch.exp(Wc['core.a_in']))
    hn = abar*htf + sp*(E @ Wc['core.b_in.weight'].T) + (x - (htf + E))
    c = hn @ Wc['core.c_out.weight'].T
    n1 = LNw(c, 'core.coda.0.norm1'); c = c + mha_m(n1, n1, Wc['core.coda.0.attn.q.weight'], Wc['core.coda.0.attn.kv.weight'], Wc['core.coda.0.attn.out.weight'])
    c = c + ffn(LNw(c, 'core.coda.0.norm3'), 'core.coda.0.')
    return c, LNw(c.mean(1), 'core.out_norm')
yt = torch.tensor(X3[rows][:, 192:960]).reshape(-1,4,192); ot = torch.tensor(X3[rows][:, :192])
dyn = (yt - htf).norm(dim=(1,2)).mean().item()
print('mean |y - htf|', round(dyn,3), '|y|', round(yt.norm(dim=(1,2)).mean().item(),3))
for incl in [True, False]:
    for unit in [3600.0, 1.0]:
        for pre in [False, True]:
            y_, o_ = step_full(rows, E0, htf, incl, unit, pre)
            ey = (y_-yt).norm(dim=(1,2)).mean().item(); eo = (o_-ot).norm(dim=1).mean().item()
            print(incl, unit, pre, 'err y', round(ey,4), 'err out', round(eo,4))
```