Still no exact match (minimum error about 1.1). ReLU with input order [ctx, e, u] is consistently best, but bundle 0 alone cannot separate errors in the initial state and gap from errors in the encoding.

Bundle 1 is a better test. It is always a single map_cuff reading at −1 h, and its context comes entirely from the known bundle-0 state (out_0 and h_0 from fwd_3, the same in every stay) after one hour of time decay. Across the 100 stays z_1 then varies only with the cuff value. Matching z_1 for all stays at once tests four things together:
- the time flow, with dt in seconds;
- the gap input;
- the measurement atom encoding: z-score, fine bin, unit index, kq;
- the activation.

An error near 1e-4 would confirm the convention. If the error stays around 1 for every option, something in the encoder is missing and I should drop the full copy.

```python
X_ = S['x']
o0 = torch.tensor(X_[first[0], :192]); h0 = torch.tensor(X_[first[0], 192:960]).reshape(4,192)
v1 = []
for k_, i in enumerate(first):
    sid = sid_[i]; rows = tu[(tu.stay_id==sid)&(tu.hour==-1.0)]
    v1.append(rows.value.values[0] if len(rows) else np.nan)
v1 = np.array(v1); print('bundle1 values sample', v1[:8], ' events', tu[(tu.stay_id==sid_[first[0]])&(tu.hour==-1.0)].event.values)
Z1 = torch.tensor(X_[first+1, 960:992])
edges = np.array(VOCAB['numeric'][1]['fine_bin_edges'])   # map_cuff
def fbin(x, edges, strict=True): return 1 + (np.sum(edges < x) if strict else np.sum(edges <= x))
res_ = []
for an in ['relu','gelu']:
  act = acts[an]
  for strict in [True, False]:
    for kq in [None,0,1,2,3]:
      for unit in range(6):
        embs = []
        for v in v1:
            num = ((np.clip((v-81)/17,-10,10)), 1.0, 0, 0, 0, 0)
            a_ = atom_emb(5, 0, 1, dict(kq=kq), vb=fbin(v, edges, strict), num=num) + (W['embed.categorical.unit.weight'][unit])
            embs.append(set_enc(a_[None], 'encode', act))
        E1 = torch.stack(embs)
        for dtu, dtv in [('s',3600.0),('h',1.0),('min',60.0)]:
          h0f = h0 * torch.exp(-3600.0*torch.exp(W['core.flow.a']))
          for gscale in ['/10','raw']:
            gin = np.log(dtv)/10 if gscale=='/10' else np.log(dtv)
            ctx = o0 + ln(h0f.mean(0),'core.out_norm') + mlp(torch.tensor([gin],dtype=torch.float32),'gap',act)
            inp = torch.cat([ctx[None].expand(100,-1), E1, torch.zeros(100,192)],1)
            zo = lin(act(lin(inp,'z_post.0')),'z_post.2')[:, :32]
            res_.append(((zo-Z1).abs().max().item(), (zo-Z1).abs().mean().item(), an, strict, kq, unit, dtu, gscale))
res_.sort(key=lambda t: t[0])
for b_ in res_[:8]: print(b_)
```