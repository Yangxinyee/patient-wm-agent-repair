The context formula is confirmed to float precision (max error 6e-4, mean 6e-5):

ctx = out_prev + out_norm(mean_slots(h_prev · exp(−dt_sec·exp(flow.a)))) + gap(log(dt_sec)/10)

Here dt is in seconds, and the decay timescales run from 615 s to 2 years, as the config says.

The last input to z_post is e_k, from the measurement encoder. I'll fit the measurement atom the same way as the drug atom:

code[c] + kind[0] + ekind[1] + unit row + bins.value[b] + numeric([z_value, 1, 0, 0, 0, 0]) + numeric.bias

The unknowns are which trained unit row (2 or 5) belongs to mg/dL and which to mmHg, and whether kq[0] is included. I'll test only plain integer-hour bundles of readings, against the probe's 6 exact e_k coordinates. An exact match (error about 1e-5) gives me a complete CPU reproduction of z_post's inputs. That is what I need to see how existing z_post hidden units behave before repurposing any.

```python
cid = {c['name']: c['id'] for c in V['codes']}
evs = tu[['stay_id','hour','event','value']].copy(); evs['hk'] = evs.hour.round(4)
gb = {k: g for k, g in evs.groupby(['stay_id','hk'])}
meas_codes = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
e_dims = [j for (j,b,c) in probe_cols if b=='e']
cand_idx = []
for i in np.where(nod)[0]:
    if abs(hr4[i] - round(hr4[i])) > 1e-6 or hr4[i] < 0.5: continue
    g = gb.get((sid4[i], round(float(hr4[i]),4)))
    if g is None or not g.event.isin(meas_codes).all(): continue
    cand_idx.append(i)
    if len(cand_idx) >= 120: break
print('test bundles', len(cand_idx))
def meas_parts(code, val):
    d = num[(code,'value')]
    return load('embed.bins.value.weight')[fbin(val, d['fine_bin_edges'])] + NW @ torch.tensor([zsc(val, d), 1., 0., 0., 0., 0.]) + NB + E_['code'][cid[code]]
def enc_e(atoms):
    A = torch.stack(atoms); n = len(atoms)
    h1 = gelu(A @ load('encode.mlp1.0.weight').T + load('encode.mlp1.0.bias')) @ load('encode.mlp1.2.weight').T + load('encode.mlp1.2.bias')
    s = h1.sum(0)/np.sqrt(n) + load('encode.size.weight')[:,0]*np.log1p(n) + load('encode.size.bias')
    h2 = gelu(load('encode.mlp2.0.weight') @ s + load('encode.mlp2.0.bias')) @ load('encode.mlp2.2.weight').T + load('encode.mlp2.2.bias')
    return LNf(h2, load('encode.norm.weight'), load('encode.norm.bias'))
bund = [[(r.event, r.value) for r in gb[(sid4[i], round(float(hr4[i]),4))].itertuples()] for i in cand_idx]
TE = torch.tensor(Z4[cand_idx][:, e_dims])
resE = []
for ug, um, kqi in itertools.product([2,5,0],[2,5,0],[None,0]):
    def atom(code, val):
        a = meas_parts(code, val) + E_['kind'][0] + E_['ekind'][1] + CAT['unit'][ug if code.startswith('glucose') else um]
        return a + (E_['kq'][kqi] if kqi is not None else 0)
    pe = torch.stack([enc_e([atom(c_, v_) for c_, v_ in b_])[e_c] for b_ in bund])
    resE.append((float((pe - TE).abs().max()), ug, um, kqi))
resE.sort(key=lambda t: t[0]); print(resE[:4])
```