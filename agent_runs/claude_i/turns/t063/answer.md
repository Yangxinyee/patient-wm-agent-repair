Every bundle's atom count matches the tool's `n_atoms` exactly (5156 bundles, about 52 per stay). This includes END atoms when each infusion record ends and for the admission rows at stay end. So the bundling is settled.

Next, I will build the atom embedding and both set encoders and test them on every bundle after the first. The test is z_post([ctx, e_k, u_k]) against the stored 32-dim posterior mean.

My first guess at the unknowns:
- unit rows: mmHg = 2, mg/dL = 3, units/hour = 4, mcg/kg/min = 5;
- rate_unit rows: units/hour = 2, mcg/kg/min = 3;
- event kinds taken from the vocabulary; END atoms of drugs carrying rate and dose.

Errors are reported separately for measurement-only, drug-only and mixed bundles, so a wrong guess shows up in the bundle type it affects. A near-exact match on measurement-only bundles validates the measurement encoder. The drug-only bundles then isolate the action encoder's unknowns.

```python
enc_code = r'''
EK = {c["name"]: {"measurement":1,"drug_admin":2,"visit":3,"visit_detail":4}.get(c.get("event_kind"),0) for c in VOCAB["codes"]}
def atom_matrix(atoms, cfg):
    W = {k: Tn(k) for k in ['embed.code.weight','embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.categorical.unit.weight',
         'embed.categorical.rate_unit.weight','embed.bins.value.weight','embed.bins.rate.weight','embed.bins.dose.weight','embed.numeric.weight','embed.numeric.bias']}
    V = np.zeros((len(atoms),192))
    for i,a in enumerate(atoms):
        c = a['code']; v = W['embed.code.weight'][CID[c]] + W['embed.kind.weight'][a['kind']] + W['embed.ekind.weight'][EK[c]] + W['embed.kq.weight'][0]
        num = np.zeros(6)
        if c in ('glucose_finger','glucose_lab','map_arterial','map_cuff'):
            v = v + W['embed.categorical.unit.weight'][cfg['unit'][c]]
            num[0] = zsc(c,'value',a['value']); num[1] = 1
            v = v + W['embed.bins.value.weight'][fbin(c,'value',np.array([a['value']]))[0]]
        elif c in ('insulin','norepinephrine'):
            v = v + W['embed.categorical.unit.weight'][cfg['unit'][c]] + W['embed.categorical.rate_unit.weight'][cfg['runit'][c]]
            if a['kind']==0 or cfg['end_attrs']:
                num[2] = zsc(c,'dose',a['dose']); num[3]=1; num[4] = zsc(c,'rate',a['rate']); num[5]=1
                v = v + W['embed.bins.dose.weight'][fbin(c,'dose',np.array([a['dose']]))[0]] + W['embed.bins.rate.weight'][fbin(c,'rate',np.array([a['rate']]))[0]]
        v = v + W['embed.numeric.weight'] @ num + W['embed.numeric.bias']
        V[i] = v
    return V
def set_encode(V, pre):
    n = V.shape[0]
    s = mlp2(V, pre + '.mlp1').sum(0) / np.sqrt(n) + Tn(pre + '.size.weight')[:,0]*np.log1p(n) + Tn(pre + '.size.bias')
    return lnorm(mlp2(s[None,:], pre + '.mlp2')[0], Tn(pre + '.norm.weight'), Tn(pre + '.norm.bias'))
def bundle_eu(atoms, cfg):
    ms = [a for a in atoms if a['code'] not in ('insulin','norepinephrine')]
    ds = [a for a in atoms if a['code'] in ('insulin','norepinephrine')]
    e = set_encode(atom_matrix(ms,cfg), 'encode') if ms else np.zeros(192)
    u = set_encode(atom_matrix(ds,cfg), 'encode_action') if ds else np.zeros(192)
    return e, u
'''
open('/work/enc.py','w').write(enc_code); exec(enc_code)
print({k:EK[k] for k in ['hospital_admission','icu_admission','insulin','glucose_finger']})
cfg0 = dict(unit={'map_cuff':2,'map_arterial':2,'glucose_finger':3,'glucose_lab':3,'insulin':4,'norepinephrine':5}, runit={'insulin':2,'norepinephrine':3}, end_attrs=True)
# previous-row index & ctx for all bundles k>=1
prev = np.r_[-1, np.arange(len(S3)-1)]; okb = np.r_[False, S3[1:]==S3[:-1]]
idxb = np.where(okb)[0]; pkb = idxb-1
dtb = (H3[idxb]-H3[pkb])*3600
Hs_ = X3[pkb,192:960].reshape(-1,4,192)*np.exp(-dtb[:,None]*np.exp(P['fa'])[None,:])[:,None,:]
ctxb = X3[pkb,:192] + lnorm(Hs_.mean(1), P['ow'], P['ob']) + mlp2(np.log(dtb)[:,None]/10.0, 'gap')
def zpost_check(cfg, sub=None):
    ii = np.arange(len(idxb)) if sub is None else sub
    E_ = []; U_ = []
    for j in ii:
        e,u = bundle_eu(atoms_by_bundle[mkey(S3[idxb[j]],H3[idxb[j]])], cfg); E_.append(e); U_.append(u)
    E_ = np.array(E_); U_ = np.array(U_)
    zo = mlp2(np.c_[ctxb[ii], E_, U_], 'z_post')
    tgt = X3[idxb[ii], 960:992]
    return zo, tgt, E_, U_
zo, tgt, E0, U0 = zpost_check(cfg0)
kinds = np.array(['meas' if (np.abs(U0[j]).sum()==0) else ('drug' if np.abs(E0[j]).sum()==0 else 'both') for j in range(len(idxb))])
for kk in ['meas','drug','both']:
    ix = kinds==kk; d_ = zo[ix,:32]-tgt[ix]
    print(kk, ix.sum(), "MAD", np.abs(d_).mean().round(4), "corr", np.corrcoef(zo[ix,:32].ravel(), tgt[ix].ravel())[0,1].round(5))
```