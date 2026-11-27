The table norms separate trained rows from untrained ones. Rows near 0.28 look like untouched initialisation:
- **Used:** kind rows 0 (event) and 2 (END); event kinds 1–4; `kq` row 0; unit rows 2–5; rate-unit rows 2–3.
- **Unused:** action, route, status, discharged_to and text. Only their padding row would ever be used, so they contribute nothing.

That leaves a small set of unknowns: which unit row is mg/dL, mmHg, mg or units; which rate-unit row is which; whether END atoms carry attributes; and what e or u is for an empty set. I'll build the atom embedding and set encoders from scratch, compute z posterior = z_post([ctx, e_k, u_k]) for every non-first bundle, and compare with the stored z for each mapping variant.
- A variant with near-zero error validates the encoders and the context, which is the basis for a core replica.
- If every variant stays far off, the encoder description needs rethinking.

```python
import itertools
nm_={ (e['code'],e['attribute']):e for e in VOCAB['numeric']}
def zs_(code,attr,x): e=nm_[(code,attr)]; return float(np.clip((x-e['median'])/e['iqr'],-10,10))
def fb_(code,attr,x): e=nm_[(code,attr)]; return int(1+np.searchsorted(np.array(e['fine_bin_edges']),x,side='left'))
CID={c['name']:c['id'] for c in VOCAB['codes']}
EK={'measurement':1,'drug_admin':2,'visit':3,'visit_detail':4}
EV2EK={c['name']:EK.get(c.get('event_kind'),0) for c in VOCAB['codes']}
# bundles per stay from tuning records
bund={}
for s_ in np.unique(sid_):
    d_=tu[tu.stay_id==s_]; atoms=[]
    for r in d_.itertuples():
        atoms.append((round(r.hour*60), dict(ev=r.event,kind=0,value=r.value,rate=r.rate,dose=r.dose,ru=r.rate_unit,du=r.dose_unit)))
        if r.end_hour==r.end_hour: atoms.append((round(r.end_hour*60), dict(ev=r.event,kind=2,value=np.nan,rate=np.nan,dose=np.nan,ru=None,du=None)))
    bb={}
    for m,a in atoms: bb.setdefault(m,[]).append(a)
    bund[s_]=[bb[m] for m in sorted(bb)]
print('bundle count check', sum(len(v) for v in bund.values()), len(sid_))
EMB={n:Wt(n) for n in ['embed.code.weight','embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.categorical.unit.weight',
     'embed.categorical.rate_unit.weight','embed.bins.value.weight','embed.bins.dose.weight','embed.bins.rate.weight','embed.numeric.weight','embed.numeric.bias']}
def atom_e(a,umap,rumap,end_attr=False):
    ev=a['ev']; v=EMB['embed.code.weight'][CID[ev]]+EMB['embed.kind.weight'][a['kind']]+EMB['embed.ekind.weight'][EV2EK[ev]]+EMB['embed.kq.weight'][0]
    num=np.zeros(6)
    unit=a['du'] if a['du'] else (VOCAB['codes'][CID[ev]].get('unit') if ev in CODE else None)
    if a['kind']==0 or end_attr:
        if unit in umap: v=v+EMB['embed.categorical.unit.weight'][umap[unit]]
        if a['ru'] in rumap: v=v+EMB['embed.categorical.rate_unit.weight'][rumap[a['ru']]]
    if a['value']==a['value'] and a['value'] is not None:
        num[0]=zs_(ev,'value',a['value']); num[1]=1; v=v+EMB['embed.bins.value.weight'][fb_(ev,'value',a['value'])]
    if a['dose']==a['dose'] and a['dose'] is not None:
        num[2]=zs_(ev,'dose',a['dose']); num[3]=1; v=v+EMB['embed.bins.dose.weight'][fb_(ev,'dose',a['dose'])]
    if a['rate']==a['rate'] and a['rate'] is not None:
        num[4]=zs_(ev,'rate',a['rate']); num[5]=1; v=v+EMB['embed.bins.rate.weight'][fb_(ev,'rate',a['rate'])]
    return v+EMB['embed.numeric.weight']@num+EMB['embed.numeric.bias']
def enc_set(vs,prefix,act):
    if len(vs)==0: return np.zeros(192)
    A=np.stack(vs); h1=M2(A,prefix+'.mlp1',act).sum(0)/np.sqrt(len(vs))
    pooled=h1+L(np.array([np.log1p(len(vs))]),prefix+'.size')
    return LNf(M2(pooled,prefix+'.mlp2',act),prefix+'.norm')
ACTION={'insulin','norepinephrine'}
units=['mg/dL','mmHg','mg','units']
def zpost_err(umap,rumap,act='gelu',end_attr=False,nmax=1500):
    errs=[];cnt=0
    for s_ in np.unique(sid_):
        ii=starts[s_]; B=bund[s_]
        if len(B)!=len(ii): continue
        for k in range(1,len(ii)):
            j0,j1=ii[k-1],ii[k]
            dtk=(hr_[j1]-hr_[j0])*3600
            hprev=X_[j0,192:960].reshape(4,192)*np.exp(-dtk*np.exp(Wt('core.flow.a')))[None,:]
            ctx=X_[j0,:192]+LNf(hprev.mean(0),'core.out_norm')+M2(np.array([np.log(dtk)/10]),'gap',act)
            ea=[atom_e(a,umap,rumap,end_attr) for a in B[k] if a['ev'] not in ACTION]
            ua=[atom_e(a,umap,rumap,end_attr) for a in B[k] if a['ev'] in ACTION]
            e_=enc_set(ea,'encode',act); u_=enc_set(ua,'encode_action',act)
            zp=M2(np.concatenate([ctx,e_,u_]),'z_post',act)[:32]
            errs.append(np.abs(zp-X_[j1,960:]).mean()/np.abs(X_[j1,960:]).mean()); cnt+=1
        if cnt>nmax: break
    return np.mean(errs), np.median(errs)
rus=['mcg/kg/min','units/hour']
out_=[]
for perm in itertools.permutations([2,3,4,5]):
    umap=dict(zip(units,perm))
    for rp in [(2,3),(3,2)]:
        rumap=dict(zip(rus,rp))
        out_.append((zpost_err(umap,rumap,nmax=150),perm,rp))
out_.sort(key=lambda r:r[0][0]); print(out_[:4])
```