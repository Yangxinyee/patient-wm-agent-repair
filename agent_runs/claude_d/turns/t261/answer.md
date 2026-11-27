The tables are small: 45 categorical, kind, code and kq rows in all, plus 51 bin rows and the numeric columns. Each drug record has exactly one rate unit and one dose unit. Two consequences:
- **Gate:** I can make the gate exact whatever the unknown ids are. The "norepinephrine" and "top-bin" detector directions get orthogonalized against every other embedding row.
- **Correction vector:** this does need the true atom vector.

So I will identify the unknown unit ids by the z-posterior check. The setup:
- Take norepinephrine START bundles that contain only that single atom.
- Compute ctx from the previous stored state (decayed slots, out_norm, gap).
- Encode the atom with candidate unit/rate_unit ids, with e_k either zero or the empty-set encoding.
- Compare z_post's mean with the stored z posterior mean.

A near-zero error for one combination confirms both my encoder and those ids. Uniformly large errors mean some term is still missing.

```python
Wo = lambda n: np.asarray(load_original(n), dtype=np.float64)
def LNo(x, n, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x-m)**2).mean(-1, keepdims=True); return (x-m)/np.sqrt(v+eps)*Wo(n+'.weight') + Wo(n+'.bias')
def Lo(x, n): return x@Wo(n+'.weight').T + Wo(n+'.bias')
def MLPo(x, n): return Lo(gel(Lo(x, n+'.0')), n+'.2')
fa = Wo('core.flow.a')
def ctx_of(p, dt):
    out = XALL[p, :192].astype(np.float64); h = XALL[p, 192:960].reshape(4,192).astype(np.float64)
    return out + LNo((h*np.exp(-dt*np.exp(fa))).mean(0), 'core.out_norm') + MLPo(np.array([[np.log(dt)/10]]), 'gap')[0]
EM = {k: Wo('embed.%s.weight'%k) for k in ['code','kind','ekind','bins.rate','bins.dose','bins.value','categorical.unit','categorical.rate_unit','categorical.route','categorical.status','categorical.action','kq','text']}
NW = Wo('embed.numeric.weight'); NB = Wo('embed.numeric.bias')
def nent(drug, att): return [d for d in nu if d['code']==drug and d['attribute']==att][0]
def zs_(drug, att, x): e = nent(drug, att); return float(np.clip((x-e['median'])/e['iqr'], -10, 10))
def fb_(drug, att, x): return int(1 + np.searchsorted(nent(drug, att)['fine_bin_edges'], x, side='left'))
def avec(code, kind, rate, dose, drug, u, ru, extra=0.0):
    return (EM['code'][code] + EM['kind'][kind] + EM['ekind'][2] + EM['bins.rate'][fb_(drug,'rate',rate)] + EM['bins.dose'][fb_(drug,'dose',dose)] + EM['bins.value'][0]
            + NW@np.array([0,0,zs_(drug,'dose',dose),1,zs_(drug,'rate',rate),1]) + NB + EM['categorical.unit'][u] + EM['categorical.rate_unit'][ru] + extra)
def uenc(atoms, n='encode_action'):
    s = sum(MLPo(a[None], n+'.mlp1')[0] for a in atoms)/np.sqrt(len(atoms)) + Lo(np.array([[np.log(1+len(atoms))]]), n+'.size')[0]
    return LNo(MLPo(s[None], n+'.mlp2')[0], n+'.norm')
e_empty = LNo(MLPo(Lo(np.array([[0.0]]), 'encode.size'), 'encode.mlp2')[0], 'encode.norm')
S400 = set(STALL.stay_id.unique()); T4 = TRAIN[TRAIN.stay_id.isin(S400)].copy(); T4['m'] = np.round(T4.hour*60).astype(int)
EE = T4[T4.end_hour.notna()].copy(); EE['m'] = np.round(EE.end_hour*60).astype(int)
cnt = pd.concat([T4[['stay_id','m']], EE[['stay_id','m']]]).groupby(['stay_id','m']).size()
SI = {(int(a),int(b)): i for i,(a,b) in enumerate(zip(STALL.stay_id.values, STALL.minute.values))}
cand = [r for r in T4[T4.event=='norepinephrine'].itertuples() if cnt.get((r.stay_id, r.m),0)==1 and (int(r.stay_id),int(r.m)) in SI][:40]
print('single NE bundles', len(cand))
rows_ = []
for r in cand:
    j = SI[(int(r.stay_id),int(r.m))]; p = int(STALL.prev_row.values[j]); dt = float((STALL.minute.values[j]-STALL.prev_min.values[j])*60)
    if p < 0 or dt <= 0: continue
    rows_.append((j, ctx_of(p, dt), r.rate, r.dose))
res_ = []
for u in range(6):
    for ru in range(4):
        uks = [uenc([avec(9, 0, rt, ds, 'norepinephrine', u, ru)]) for (_,_,rt,ds) in rows_]
        for en, ek in [('zero', np.zeros(192)), ('empty', e_empty)]:
            err = [np.abs(MLPo(np.r_[c, ek, uk][None], 'z_post')[0][:32] - XALL[j, 960:992]).mean() for (j,c,_,_), uk in zip(rows_, uks)]
            res_.append((float(np.mean(err)), u, ru, en))
res_.sort(); print(len(rows_), [tuple(np.round(x,4) if isinstance(x,float) else x for x in t) for t in res_[:5]], 'worst', round(res_[-1][0],4), 'z scale', round(float(np.abs(XALL[[r[0] for r in rows_], 960:992]).mean()),3))
```