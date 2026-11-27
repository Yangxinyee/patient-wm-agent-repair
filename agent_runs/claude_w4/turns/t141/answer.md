I have the encode_action tensors, so I'll compute the no-action u_k as LN(mlp2(size.bias)).

Then I'll run the prelude replica on a sample of non-drug bundles. These are reading bundles at integer hours, with e_k built approximately from their measurements via `meas_parts` and `enc_e`. The question is where heads 1–3 put their attention when no drug is present. If it stays on the u_k token (now the constant no-action vector), a gate on the u_k value path is clean. If it moves to e_k, the gate must also be robust to reading content.

```python
W_ea = {n: load_original(n) for n in ea}
def mlp2_(x):
    hdn = torch.nn.functional.gelu(x @ W_ea['encode_action.mlp2.0.weight'].T + W_ea['encode_action.mlp2.0.bias'])
    return hdn @ W_ea['encode_action.mlp2.2.weight'].T + W_ea['encode_action.mlp2.2.bias']
u_const = LNt(mlp2_(W_ea['encode_action.size.bias'][None]), W_ea['encode_action.norm.weight'], W_ea['encode_action.norm.bias'])[0]
print('u_const norm', float(u_const.norm()), 'uk drug norm mean', float(uk.norm(dim=1).mean()))
mcodes = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
tus = tu[tu.stay_id.isin(np.unique(st))]
nd_rows = np.where(~dr)[0]
nd_rows = nd_rows[(nd_rows > 0)]
nd_rows = nd_rows[st[nd_rows-1] == st[nd_rows]]
rng = np.random.default_rng(0)
samp = rng.choice(nd_rows, 600, replace=False)
grp = {k: g for k, g in tus.groupby(['stay_id','hour'])}
E_nd, keep, kinds = [], [], []
for i in samp:
    g = grp.get((st[i], hr[i]))
    if g is None: continue
    gm = g[g.event.isin(mcodes)]
    if len(gm) == len(g) and len(gm) > 0:
        atoms = [meas_parts(r.event, float(r.value)) for r in gm.itertuples()]
        E_nd.append(enc_e(atoms).reshape(-1)); keep.append(i); kinds.append('meas')
print('built', len(keep))
```