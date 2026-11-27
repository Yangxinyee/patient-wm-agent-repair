The introspection shows how the old exact pipeline worked:
- Atom vectors are a plain sum of the documented embedding tables (`atom_vec`).
- The set encoder uses `.size`, `.mlp1`, `.mlp2` and `.norm`, with special handling for empty sets.
- The latent check builds ctx from the previous saved state: time-flow decay with dt in seconds (×3600), out_norm, and gap(log dt / 10).

There is one conflict. The live `rumap` maps mcg/kg/min→2 and units/hour→3, the reverse of my notebook. The global may be left over from an earlier test.

Plan for this run, using only the documented formulas:
1. Rebuild the bundles for the 100 setA stays: same-minute grouping; drug START atoms to the action set; END atoms and measurements to the measurement set.
2. Check the bundle counts and atom counts against the tool's saved states.
3. Verify the latent posterior means for bundles k ≥ 1, using the saved previous state.

I test both rate-unit orders and both empty-action encodings. A relative error near 0.01 for one variant confirms the rebuild, and its stored e_k, u_k and dt can then feed the core replica. Larger errors for all variants mean a detail is wrong, and the per-type errors will show which bundle type is off.

```python
import numpy as np
from scipy.special import erf
Wo = {n: load_original(n).numpy().astype(np.float64) for n in NAMES}
def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def LNn(x, n, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x-m)**2).mean(-1, keepdims=True)
    return (x-m)/np.sqrt(v+eps)*Wo[n+'.weight'] + Wo[n+'.bias']
def linn(x, n):
    y = x @ Wo[n+'.weight'].T
    return y + Wo[n+'.bias'] if (n+'.bias') in Wo else y
def mlp2n(x, n): return linn(gelu(linn(x, n+'.0')), n+'.2')
CID = {c['name']: c['id'] for c in VOCAB['codes']}
NUM = {(d['code'], d['attribute']): d for d in VOCAB['numeric']}
def zb(code, attr, x):
    d = NUM[(code, attr)]
    return float(np.clip((x-d['median'])/d['iqr'], -10, 10)), 1+int(np.searchsorted(np.array(d['fine_bin_edges'], float), x, side='left'))
def avec(code, kind, ekind, unit=0, ru=0, vb=0, db=0, rb=0, num=(0,)*6):
    W = Wo
    return (W['embed.code.weight'][code]+W['embed.kind.weight'][kind]+W['embed.ekind.weight'][ekind]+W['embed.kq.weight'][0]
            +W['embed.categorical.unit.weight'][unit]+W['embed.categorical.rate_unit.weight'][ru]
            +W['embed.bins.value.weight'][vb]+W['embed.bins.dose.weight'][db]+W['embed.bins.rate.weight'][rb]
            +W['embed.numeric.weight']@np.array(num,float)+W['embed.numeric.bias'])
MUNIT = {'map_arterial':2,'map_cuff':2,'glucose_lab':5,'glucose_finger':0}
DUNIT = {'insulin':3,'norepinephrine':4}
def atoms_of_stay(df, RU, dscale=None):
    out = []  # (minute, cls, vec, code)
    for r in df.itertuples():
        ev = r.event
        if ev in MUNIT:
            z, b = zb(ev,'value',r.value)
            out.append((round(r.hour*60),'M',avec(CID[ev],0,1,MUNIT[ev],0,b,0,0,(z,1,0,0,0,0)),ev))
        elif ev in DUNIT:
            f = 1.0 if dscale is None else dscale(ev, r.hour)
            zd, bd = zb(ev,'dose',r.dose*f); zr, br = zb(ev,'rate',r.rate*f)
            out.append((round(r.hour*60),'A',avec(CID[ev],0,2,DUNIT[ev],RU[r.rate_unit],0,bd,br,(0,0,zd,1,zr,1)),ev))
            out.append((round(r.end_hour*60),'M',avec(CID[ev],2,2),ev+'_END'))
        else:
            ek = 3 if ev=='hospital_admission' else 4
            out.append((round(r.hour*60),'M',avec(CID[ev],0,ek),ev))
            if not np.isnan(r.end_hour): out.append((round(r.end_hour*60),'M',avec(CID[ev],2,ek),ev+'_END'))
    return out
def encset(vs, prefix, empty='size'):
    if len(vs)==0:
        if empty=='zero': return np.zeros(192)
        pooled = linn(np.array([0.0]), prefix+'.size')
    else:
        A = np.stack(vs); pooled = mlp2n(A, prefix+'.mlp1').sum(0)/np.sqrt(len(vs)) + linn(np.array([np.log1p(len(vs))]), prefix+'.size')
    return LNn(mlp2n(pooled, prefix+'.mlp2'), prefix+'.norm')
def bundles_of_stay(df, RU, uempty='size', dscale=None):
    at = atoms_of_stay(df.sort_values('hour'), RU, dscale)
    mins = sorted(set(a[0] for a in at)); B = []
    for m in mins:
        M_ = [a[2] for a in at if a[0]==m and a[1]=='M']; A_ = [a[2] for a in at if a[0]==m and a[1]=='A']
        B.append(dict(t=m/60.0, nM=len(M_), nA=len(A_), e=encset(M_,'encode'), u=encset(A_,'encode_action',uempty),
                      labels=[a[3] for a in at if a[0]==m]))
    return B
setA_ids = json.load(open('/work/setA.json')); setA_ids = setA_ids if isinstance(setA_ids, list) else list(setA_ids.values())[0]
X3, sid3, hr3 = S3['x'].astype(np.float64), S3['stay_id'], S3['hour']
def zpost_of(ctx, e, u): return linn(gelu(linn(np.concatenate([ctx,e,u]),'z_post.0')),'z_post.2')[:32]
def ctx_of(prev, dt):
    h = prev[192:960].reshape(4,192)*np.exp(-dt*np.exp(Wo['core.flow.a']))
    return prev[:192] + LNn(h.mean(0),'core.out_norm') + mlp2n(np.array([np.log(dt)/10]),'gap')
tuA = tu[tu.stay_id.isin(setA_ids)]
res = {}
for RUname, RU in [('nb', {'units/hour':2,'mcg/kg/min':3}), ('sw', {'units/hour':3,'mcg/kg/min':2})]:
  for uempty in ['size','zero']:
    errs=[]; cnt_ok=0; cnt=0
    for s in setA_ids[:25]:
        B = bundles_of_stay(tuA[tuA.stay_id==s], RU, uempty)
        rows = np.where(sid3==s)[0]
        cnt += 1; cnt_ok += int(len(rows)==len(B) and np.allclose(hr3[rows], [b['t'] for b in B], atol=1e-3))
        if len(rows)!=len(B): continue
        for k in range(1,len(B)):
            dt = (B[k]['t']-B[k-1]['t'])*3600
            zp = zpost_of(ctx_of(X3[rows[k-1]], dt), B[k]['e'], B[k]['u'])
            zt = X3[rows[k],960:992]
            errs.append((np.abs(zp-zt).mean()/np.abs(zt).mean(), B[k]['nA']>0))
    errs = np.array(errs)
    res[(RUname,uempty)] = errs
    print(RUname, uempty, 'aligned', cnt_ok, '/', cnt, 'relerr all %.4f  drug %.4f  nondrug %.4f' % (errs[:,0].mean(), errs[errs[:,1]==1,0].mean(), errs[errs[:,1]==0,0].mean()))
```