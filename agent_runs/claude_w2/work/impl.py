
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

def encode_bundles(A,W,end_attr=False):
    A=A.copy()
    if not end_attr: A.loc[A.kind==2,['rate','dose']]=np.nan
    feats=atom_features(A); xa=embed(A,feats,W)
    key=A.stay_id.values.astype(np.int64)*100000+A.bidx.values
    ukey,gid=np.unique(key,return_inverse=True)
    nb=len(ukey)
    isact=((A.kind.values==0)&(A.ekind.values==2))
    E,ce=set_encode(xa[torch.as_tensor(~isact)],gid[~isact],nb,W,'encode')
    U,cu=set_encode(xa[torch.as_tensor(isact)],gid[isact],nb,W,'encode_action')
    B=pd.DataFrame({'stay_id':ukey//100000,'bidx':ukey%100000})
    B['minute']=A.groupby(key).minute.first().reindex(ukey).values
    B['t']=A.groupby(key).t.first().reindex(ukey).values
    return B,E,U,(cu>0).numpy(),xa

UMAP={'map_cuff':2,'glucose_finger':0,'map_arterial':2,'glucose_lab':5,'insulin':3,'norepinephrine':4}
RUMAP={'insulin':2,'norepinephrine':3}
UROW=np.zeros(13,int); RROW=np.zeros(13,int)
for _k,_v in UMAP.items(): UROW[CODE[_k]]=_v
for _k,_v in RUMAP.items(): RROW[CODE[_k]]=_v
def embed(A,feats,W):
    num,vb,db,rb=feats
    T=lambda a: torch.as_tensor(np.asarray(a),dtype=torch.long)
    c=A.code.values; isend=(A.kind.values==2)
    ur=np.where(isend,0,UROW[c]); rr=np.where(isend,0,RROW[c])
    x=(W['embed.code.weight'][T(c)]+W['embed.kind.weight'][T(A.kind.values)]+W['embed.ekind.weight'][T(A.ekind.values)]
       +W['embed.categorical.unit.weight'][T(ur)]+W['embed.categorical.rate_unit.weight'][T(rr)]+W['embed.kq.weight'][0]
       +W['embed.bins.value.weight'][T(vb)]+W['embed.bins.dose.weight'][T(db)]+W['embed.bins.rate.weight'][T(rb)]
       +torch.as_tensor(num)@W['embed.numeric.weight'].T+W['embed.numeric.bias'])
    return x

def attn_(xq,xkv,W,pre,mask=None,kadd=None,inadd=None):
    q=F.linear(xq,W[pre+'.q.weight'])
    src=xkv if inadd is None else xkv+inadd
    kv=F.linear(src,W[pre+'.kv.weight']); k,v=kv[...,:192],kv[...,192:]
    if kadd is not None: k=k+kadd
    B,Lq,_=q.shape; Lk=k.shape[1]
    q=q.reshape(B,Lq,4,48).transpose(1,2); k=k.reshape(B,Lk,4,48).transpose(1,2); v=v.reshape(B,Lk,4,48).transpose(1,2)
    a=(q@k.transpose(-1,-2))/math.sqrt(48)
    if mask is not None: a=a.masked_fill(~mask[:,None,None,:],float('-inf'))
    o=(a.softmax(-1)@v).transpose(1,2).reshape(B,Lq,192)
    return F.linear(o,W[pre+'.out.weight'])
def layer_sa(x,W,pre):
    xn=LN(x,W,pre+'.norm1'); x=x+attn_(xn,xn,W,pre+'.attn')
    return x+mlp(LN(x,W,pre+'.norm3'),W,pre+'.ffn')
def layer_blk(x,W,pre,M,mask,agee,age_mode):
    xn=LN(x,W,pre+'.norm1'); x=x+attn_(xn,xn,W,pre+'.attn')
    xn=LN(x,W,pre+'.norm2')
    if age_mode=='kadd': x=x+attn_(xn,M,W,pre+'.cross',mask=mask,kadd=agee)
    elif age_mode=='inadd': x=x+attn_(xn,M,W,pre+'.cross',mask=mask,inadd=agee)
    elif age_mode=='kin':
        q_=None; Wk=W[pre+'.cross.kv.weight']
        k=F.linear(M+agee,Wk[:192]); v=F.linear(M,Wk[192:])
        x=x+attn_kv(xn,k,v,W,pre+'.cross',mask)
    return x+mlp(LN(x,W,pre+'.norm3'),W,pre+'.ffn')
def attn_kv(xq,k,v,W,pre,mask):
    q=F.linear(xq,W[pre+'.q.weight']); B,Lq,_=q.shape; Lk=k.shape[1]
    q=q.reshape(B,Lq,4,48).transpose(1,2); k=k.reshape(B,Lk,4,48).transpose(1,2); v=v.reshape(B,Lk,4,48).transpose(1,2)
    a=(q@k.transpose(-1,-2))/math.sqrt(48)
    if mask is not None: a=a.masked_fill(~mask[:,None,None,:],float('-inf'))
    return F.linear((a.softmax(-1)@v).transpose(1,2).reshape(B,Lq,192),W[pre+'.out.weight'])
def core_step(h,out,e,u,dt,M,Mmask,Mage,W,age_mode='kadd',age_unit=1.0,age_div=1.0,z=None):
    hf=h*torch.exp(-dt[:,None,None]*torch.exp(W['core.flow.a']))
    ctx=out+LN(hf.mean(1),W,'core.out_norm')+mlp(torch.log(dt)[:,None]/10,W,'gap')
    if z is None: z=mlp(torch.cat([ctx,e,u],-1),W,'z_post')[:,:32]
    X=torch.cat([hf+W['core.slots'][None],e[:,None],u[:,None],F.linear(z,W['z_in.weight'],W['z_in.bias'])[:,None]],1)
    X=layer_sa(X,W,'core.prelude.0'); E=LN(X[:,:4],W,'core.prelude_norm')
    agee=mlp(torch.log1p(Mage/age_unit)[...,None]/age_div,W,'core.age')
    x=hf+E
    for l in ['core.blocks.0.0','core.blocks.0.1']: x=layer_blk(x,W,l,M,Mmask,agee,age_mode)
    sp=F.softplus(W['core.delta_in']); abar=torch.exp(-sp*torch.exp(W['core.a_in']))
    hn=abar*hf+sp*F.linear(E,W['core.b_in.weight'])+(x-(hf+E))
    y=layer_sa(F.linear(hn,W['core.c_out.weight']),W,'core.coda.0')
    return y,LN(y.mean(1),W,'core.out_norm'),z,ctx,hf
