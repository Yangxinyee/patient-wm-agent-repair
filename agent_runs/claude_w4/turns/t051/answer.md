Bundle 1 is not always a single map_cuff reading. Values like 246 and 230 are glucose, so for some stays the reading at −1 h is a glucose_finger or there are several atoms. My test encoded every stay as map_cuff, which invalidates it. A mean error of 0.13 with a maximum of 1.8 fits this: most stays close, the mis-encoded ones far off.

I will write a proper tokeniser for measurement bundles: code from the event name, the per-code z-score and fine bin, and ekind 1. I will compare z_1 per stay under the same hypotheses, now including the unit index per measurement family. If the correctly encoded stays reach errors near 1e-4, the encoder conventions are confirmed. If the error stays around 0.1, there is still a missing component in the embedding.

```python
code_id = {c['name']: c['id'] for c in VOCAB['codes']}
numinfo = {(e_['code'], e_['attribute']): e_ for e_ in VOCAB['numeric']}
def meas_atom(ev, v, opts, unit_idx):
    ni = numinfo[(ev,'value')]; ed = np.array(ni['fine_bin_edges'])
    zv = float(np.clip((v-ni['median'])/ni['iqr'], -10, 10))
    vb = 1 + int(np.sum(ed < v)) if opts.get('strict', True) else 1 + int(np.sum(ed <= v))
    a_ = atom_emb(code_id[ev], 0, 1, opts, vb=vb, num=(zv,1.0,0,0,0,0))
    u = unit_idx.get(VOCAB['codes'][code_id[ev]]['unit'], 0)
    return a_ + W['embed.categorical.unit.weight'][u]
b1 = []
for i in first:
    sid = sid_[i]; rows = tu[(tu.stay_id==sid)&(tu.hour==-1.0)]
    b1.append(list(zip(rows.event.values, rows.value.values)))
print('bundle1 compositions:', pd.Series([tuple(sorted(e for e,_ in b)) for b in b1]).value_counts().to_dict())
h0f = h0 * torch.exp(-3600.0*torch.exp(W['core.flow.a']))
res_ = []
for an in ['relu','gelu']:
  act = acts[an]
  ctx = o0 + ln(h0f.mean(0),'core.out_norm') + mlp(torch.tensor([np.log(3600.0)/10],dtype=torch.float32),'gap',act)
  for kq in [None,0,1,2,3]:
    for um, ug in itertools.product(range(6), range(6)):
      if um==ug: continue
      unit_idx = {'mmHg': um, 'mg/dL': ug}
      E1 = torch.stack([set_enc(torch.stack([meas_atom(ev, v, dict(kq=kq), unit_idx) for ev, v in b]), 'encode', act) for b in b1])
      inp = torch.cat([ctx[None].expand(100,-1), E1, torch.zeros(100,192)],1)
      zo = lin(act(lin(inp,'z_post.0')),'z_post.2')[:, :32]
      err = (zo-Z1).abs()
      res_.append((err.max().item(), err.mean().item(), an, kq, um, ug))
res_.sort(key=lambda t: t[0])
for b_ in res_[:6]: print(b_)
```