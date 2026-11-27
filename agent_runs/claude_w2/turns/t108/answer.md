embed.kq contributes its row 0 to every atom, a constant, and my stored atom vectors `xa` match `use_kq=True` exactly. So the full atom embedding is:

code + kind + event kind + unit row + rate-unit row + kq[0] + value, dose and rate bins + numeric projection.

Now I write a clean, file-backed implementation (/work/impl.py) of the atom builder, features, embedding and the two set encoders. I then check it in three ways:
1. the atom table against ATM;
2. the atom vectors against `xa`;
3. the z-posterior mean against the tool's recorded z (fwd_3 states), by bundle type, using the previous recorded state as context.

I compare two gap activations (GELU/SiLU) and zero versus empty-set u for bundles without actions. Errors near the earlier best (about 0.01 relative) confirm the encoder, and that is the starting point for implementing the core step next.

```python
IMPL = r'''
import numpy as np, torch, torch.nn.functional as F, json, math, pandas as pd
torch.set_num_threads(8)
V=json.load(open('/model/vocab.json'))
CODE={c['name']:c['id'] for c in V['codes']}
CNAME={v:k for k,v in CODE.items()}
NUMD={(n['code'],n['attribute']):n for n in V['numeric']}
UMAP={'map_cuff':2,'glucose_finger':3,'map_arterial':2,'glucose_lab':3,'insulin':4,'norepinephrine':5}
RUMAP={'insulin':2,'norepinephrine':3}
MEAS=('glucose_finger','glucose_lab','map_arterial','map_cuff'); DRUG=('insulin','norepinephrine')
UROW=np.zeros(13,int); RROW=np.zeros(13,int)
for _k,_v in UMAP.items(): UROW[CODE[_k]]=_v
for _k,_v in RUMAP.items(): RROW[CODE[_k]]=_v
CEN=np.array(V['value_bin_centres'],dtype=np.float64) if not isinstance(V['value_bin_centres'],dict) else None

def getW(loader):
    return {n: torch.as_tensor(np.asarray(loader(n),dtype=np.float32)) for n in NAMES}

def make_atoms(df):
    rows=[]
    for r in df[['stay_id','hour','event','value','rate','dose','end_hour']].itertuples(index=False):
        sid,h,ev,val,rt,ds,eh=r
        if ev in MEAS: rows.append((sid,h,CODE[ev],0,1,val,np.nan,np.nan))
        elif ev in DRUG:
            rows.append((sid,h,CODE[ev],0,2,np.nan,rt,ds)); rows.append((sid,eh,CODE[ev],2,2,np.nan,rt,ds))
        elif ev in ('hospital_admission','icu_admission'):
            ek=3 if ev=='hospital_admission' else 4
            rows.append((sid,h,CODE[ev],0,ek,np.nan,np.nan,np.nan)); rows.append((sid,eh,CODE[ev],2,ek,np.nan,np.nan,np.nan))
    A=pd.DataFrame(rows,columns=['stay_id','t','code','kind','ekind','value','rate','dose'])
    A['minute']=np.round(A.t.values*60).astype(int)
    A=A.sort_values(['stay_id','minute','kind','code'],kind='stable').reset_index(drop=True)
    A['bidx']=A.groupby('stay_id').minute.transform(lambda m: pd.factorize(m)[0]).astype(int)
    return A

def atom_features(A):
    n=len(A); num=np.zeros((n,6),np.float32); bins={}
    codes=A.code.values
    for attr,(zc,hc) in [('value',(0,1)),('dose',(2,3)),('rate',(4,5))]:
        x=A[attr].values.astype(float); bb=np.zeros(n,np.int64)
        for cid in np.unique(codes):
            key=(CNAME[cid],attr)
            if key not in NUMD: continue
            m=(codes==cid)&~np.isnan(x); nd=NUMD[key]
            num[m,zc]=np.clip((x[m]-nd['median'])/nd['iqr'],-10,10); num[m,hc]=1
            bb[m]=1+np.searchsorted(np.array(nd['fine_bin_edges']),x[m],side='left')
        bins[attr]=bb
    return num,bins['value'],bins['dose'],bins['rate']

def embed(A,feats,W):
    num,vb,db,rb=feats
    T=lambda a: torch.as_tensor(np.asarray(a),dtype=torch.long)
    c=A.code.values
    x=(W['embed.code.weight'][T(c)]+W['embed.kind.weight'][T(A.kind.values)]+W['embed.ekind.weight'][T(A.ekind.values)]
       +W['embed.categorical.unit.weight'][T(UROW[c])]+W['embed.categorical.rate_unit.weight'][T(RROW[c])]+W['embed.kq.weight'][0]
       +W['embed.bins.value.weight'][T(vb)]+W['embed.bins.dose.weight'][T(db)]+W['embed.bins.rate.weight'][T(rb)]
       +torch.as_tensor(num)@W['embed.numeric.weight'].T+W['embed.numeric.bias'])
    return x

ACTF={'gelu':F.gelu,'silu':F.silu}
def mlp(x,W,pre,act='gelu'): return F.linear(ACTF[act](F.linear(x,W[pre+'.0.weight'],W[pre+'.0.bias'])),W[pre+'.2.weight'],W[pre+'.2.bias'])
def LN(x,W,pre,eps=1e-5): return F.layer_norm(x,(x.shape[-1],),W[pre+'.weight'],W[pre+'.bias'],eps)

def set_encode(xa,gidx,ng,W,pre):
    gidx=torch.as_tensor(np.asarray(gidx),dtype=torch.long)
    h=mlp(xa,W,pre+'.mlp1')
    s=torch.zeros(ng,xa.shape[1]).index_add_(0,gidx,h)
    cnt=torch.zeros(ng).index_add_(0,gidx,torch.ones(len(gidx)))
    pooled=s/torch.sqrt(cnt.clamp(min=1))[:,None]+F.linear(torch.log1p(cnt)[:,None],W[pre+'.size.weight'],W[pre+'.size.bias'])
    return LN(mlp(pooled,W,pre+'.mlp2'),W,pre+'.norm'),cnt

def encode_bundles(A,W,u_none='zero'):
    # returns per-bundle table (stay_id,bidx,minute), E (nb,192), U (nb,192), has_u
    feats=atom_features(A); xa=embed(A,feats,W)
    key=A.stay_id.values.astype(np.int64)*100000+A.bidx.values
    ukey,gid=np.unique(key,return_inverse=True)   # sorted by stay then bidx
    nb=len(ukey)
    isact=((A.kind.values==0)&(A.ekind.values==2))
    E,_=set_encode(xa[torch.as_tensor(~isact)],gid[~isact],nb,W,'encode')
    U,cu=set_encode(xa[torch.as_tensor(isact)],gid[isact],nb,W,'encode_action')
    hasu=(cu>0)
    if u_none=='zero': U=U*hasu[:,None].float()
    B=pd.DataFrame({'stay_id':ukey//100000,'bidx':ukey%100000})
    B['minute']=A.groupby(key).minute.first().reindex(ukey).values
    return B,E,U,hasu.numpy(),xa
'''
open('/work/impl.py','w').write(IMPL)
exec(IMPL)
WO=getW(load_original)
A0=make_atoms(tr[tr.stay_id.isin(selT)]) if 'stay_id' in tr.columns else None
print(len(A0), len(ATM), (A0[['t','code','kind','ekind']].values==ATM.sort_values(['stay_id','minute','kind','code'],kind='stable')[['t','code','kind','ekind']].values).mean() if len(A0)==len(ATM) else 'len mismatch')
```