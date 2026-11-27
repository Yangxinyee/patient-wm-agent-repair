Probing q for the random part of the insulin dose failed. The previous dose is well predicted from q (cross-validated R² 0.73), but almost all of that is the dose the glucose readings already imply. The part beyond the readings is not linearly decodable at all (R² −0.36).

So the model's state barely carries the dose information that matters, which explains its weak dose response. A correction that only reads q cannot work. The dose has to be carried more strongly through the core: encode_action → prelude → recurrent block → coda → out.

Designing that requires computing the core myself. I will rebuild the model step by step in numpy and check each stage against the saved states. The z posterior mean in the states gives an intermediate check: it is a known function of the context (from the previous state) and the bundle encodings e_k and u_k.

I start with reading-only bundles, where u_k = 0. This checks the atom embedding and the measurement set encoder. The mapping from units to rows of the unit table is unknown, so I search it.

If some variant reproduces the saved posterior means almost exactly (error near float precision), the encoder is right. If the error stays large, I am missing an embedding term.

```python
def atom_vec(code, kind=0, ekind=1, unit_row=None, vbin=0, zv=0.0, hv=0.0, zd=0.0, hd=0.0, zr=0.0, hr=0.0, dbin=0, rbin=0, extra=None):
    v_ = W['embed.code.weight'][code] + W['embed.kind.weight'][kind] + W['embed.ekind.weight'][ekind]
    if unit_row is not None: v_ = v_ + W['embed.categorical.unit.weight'][unit_row]
    v_ = v_ + W['embed.bins.value.weight'][vbin] + W['embed.bins.dose.weight'][dbin] + W['embed.bins.rate.weight'][rbin]
    v_ = v_ + W['embed.numeric.weight'] @ np.array([zv, hv, zd, hd, zr, hr]) + W['embed.numeric.bias']
    if extra is not None: v_ = v_ + extra
    return v_
def set_enc(vecs, p):
    n_ = len(vecs); s_ = sum(mlp(v_, p + '.mlp1') for v_ in vecs) / np.sqrt(n_)
    return ln(mlp(s_ + lin(np.array([np.log1p(n_)]), p + '.size'), p + '.mlp2'), W[p + '.norm.weight'], W[p + '.norm.bias'])
def ctx_of(row_prev, dt):
    x_ = X[row_prev].astype(np.float64); out = x_[:192]; h = x_[192:960].reshape(4, 192)
    hf = h * np.exp(-dt * flow_k)
    return out + ln(hf.mean(0), W['core.out_norm.weight'], W['core.out_norm.bias']) + mlp(np.array([np.log(dt) * 0.1]), 'gap')
trS = tr[tr.stay_id.isin(stays99)]
ev_by = {k_: g_ for k_, g_ in trS.groupby(['stay_id', 'hour'])}
cand = []
for idx_ in st_s.index[:1500]:
    r_ = st.loc[idx_]
    if r_.hi or r_.hn or r_.hour < 0: continue
    g_ = ev_by.get((r_.stay, r_.hour));
    if g_ is None or not g_.event.isin(list(CEN)).all(): continue
    prev_rows = st_s[(st_s.stay == r_.stay) & (st_s.hour < r_.hour)]
    if len(prev_rows) == 0: continue
    cand.append((idx_, prev_rows.index[-1], (r_.hour - prev_rows.hour.values[-1]) * 3600, g_))
    if len(cand) >= 40: break
print('n bundles', len(cand))
def zpost_err(unit_map):
    errs = []
    for idx_, ip, dt, g_ in cand:
        vecs = [atom_vec(CODE[e_], unit_row=unit_map.get(e_), vbin=fbin(e_, 'value', v_), zv=zsc(e_, 'value', v_), hv=1.0) for e_, v_ in zip(g_.event, g_.value)]
        e_k = set_enc(vecs, 'encode')
        zp = mlp(np.concatenate([ctx_of(ip, dt), e_k, np.zeros(192)]), 'z_post')[:32]
        errs.append(np.abs(zp - X[idx_, 960:992]).mean())
    return np.mean(errs)
print('no unit table:', zpost_err({}))
best = []
for ug in range(6):
    for um in range(6):
        um_ = {'glucose_finger': ug, 'glucose_lab': ug, 'map_arterial': um, 'map_cuff': um}
        best.append((zpost_err(um_), ug, um))
best.sort(); print(best[:5]); print('saved z abs mean', np.abs(X[:, 960:992]).mean())
```