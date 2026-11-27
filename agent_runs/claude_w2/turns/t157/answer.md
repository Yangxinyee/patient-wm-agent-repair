The first bundle of a stay is also exact: h = 0, out = 0, and gap computed with dt = 1 s (log dt = 0). I now have every piece of the model locally.

Next I'll write the local dose_edit rollout engine (/work/rollout.py). It steps all paths and arms of all stays bundle by bundle, as the tool describes:
- readings before hour 13 are teacher-forced; from hour 13 values are drawn from the 4-draw-averaged head probabilities and fed back as bin centres;
- drug atoms from hour 12 on carry the edited rate and dose;
- random numbers are common across arms;
- memory keys and values are cached per entry.

I'll test it on 30 sel1 stays and compare the four mean shifts and the factual predicted means with the tool's original-model output. If the engine is faithful, the shifts should come out near the tool's (−2.1, +1.3, +1.1, −0.7) within path noise, and the factual predicted means should correlate closely.

```python
code_roll = r'''
import numpy as np, torch, torch.nn.functional as F, math, pandas as pd
GLU=('glucose_finger','glucose_lab'); 
def _cen_table():
    cm=np.zeros((13,16)); 
    for nm,arr in V['value_bin_centres'].items(): cm[CODE[nm]]=np.array(arr)
    return cm
CENT=_cen_table()
def _z(code,x):
    nd=NUMD[(CNAME[code],'value')]; return np.clip((x-nd['median'])/nd['iqr'],-10,10)
def prep_rollout(df, ids):
    A=make_atoms(df[df.stay_id.isin(ids)])
    stays=np.array(sorted(A.stay_id.unique())); s_of={s:i for i,s in enumerate(stays)}
    A['si']=A.stay_id.map(s_of).values
    A=A.sort_values(['si','minute','kind','code'],kind='stable').reset_index(drop=True)
    A.loc[A.kind==2,['rate','dose']]=np.nan
    nb=A.groupby('si').bidx.max().values+1; Kmax=int(nb.max())
    tb=np.full((len(stays),Kmax),np.nan); g=A.groupby(['si','bidx']).t.first()
    tb[g.index.get_level_values(0),g.index.get_level_values(1)]=g.values
    A['loc']=A.groupby('si').cumcount().values
    atoms_sk={}
    for (si,k),idx in A.groupby(['si','bidx']).indices.items(): atoms_sk[(si,k)]=np.asarray(idx)
    return dict(A=A,stays=stays,nb=nb,Kmax=Kmax,tb=tb,atoms_sk=atoms_sk,maxloc=int(A['loc'].max())+1)
def arm_embed(R,W,arm):
    A=R['A'].copy()
    for drug,key in [('insulin','ins'),('norepinephrine','ne')]:
        f,h0=arm.get(key,1.0),arm.get(key+'_from',12.0)
        m=(A.code==CODE[drug])&(A.kind==0)&(A.t>=h0)
        if arm.get(key+'_drop',False) and f==0: 
            A.loc[m,['rate','dose']]=np.nan
        else: A.loc[m,'rate']*=f; A.loc[m,'dose']*=f
    return embed(A,atom_features(A),W)
def run_rollout(R,W,arms,n_paths=20,seed=0,t_hide=13.0,want_nll=False,chunk=None):
    A=R['A']; S_=len(R['stays']); nA=len(arms); Kmax=R['Kmax']; nb=R['nb']; tb=R['tb']
    code=A.code.values; kind=A.kind.values; ek=A.ekind.values; tt=A.t.values; val=A.value.values
    ismeas=np.isin(code,[CODE[c] for c in MEAS]); hidden=ismeas&(tt>=t_hide)
    isact=(kind==0)&(ek==2)
    X=torch.stack([arm_embed(R,W,a) for a in arms])          # nA x natoms x 192
    A0=A.copy(); A0.loc[hidden,'value']=np.nan
    Xnv=embed(A0,atom_features(A0),W)
    Wnum=W['embed.numeric.weight']
    vtab=torch.zeros(13,17,192)
    for c in [CODE[m] for m in MEAS]:
        zc=torch.as_tensor(_z(c,CENT[c]),dtype=torch.float32)
        vtab[c,1:]=W['embed.bins.value.weight'][1:17]+zc[:,None]*Wnum[:,0][None]+Wnum[:,1][None]
    # factual register values for non-hidden meas atoms
    zfact=np.zeros(len(A)); bfact=np.zeros(len(A),int)
    for c in [CODE[m] for m in MEAS]:
        m=(code==c)&ismeas&~np.isnan(val)
        zfact[m]=_z(c,val[m]); bfact[m]=1+np.searchsorted(np.array(NUMD[(CNAME[c],'value')]['fine_bin_edges']),val[m],side='left')
    # sequences
    seq_si=np.repeat(np.arange(S_),nA*n_paths); seq_arm=np.tile(np.repeat(np.arange(nA),n_paths),S_); seq_p=np.tile(np.arange(n_paths),S_*nA)
    N=len(seq_si)
    rng=np.random.default_rng(seed)
    EPS=torch.as_tensor(rng.standard_normal((S_,n_paths,Kmax,4,32)),dtype=torch.float32)
    UNI=rng.random((S_,n_paths,R['maxloc']))
    h=torch.zeros(N,4,192); out=torch.zeros(N,192)
    L=['core.blocks.0.0','core.blocks.0.1']
    Mk={l:torch.zeros(N,Kmax,192) for l in L}; Mv={l:torch.zeros(N,Kmax,192) for l in L}
    reg_has=np.zeros((N,13),bool); reg_t=np.zeros((N,13)); reg_z=np.zeros((N,13)); reg_b=np.zeros((N,13),int)
    drawn=np.zeros((N,R['maxloc']),int)
    pm_rec={}; nll_rec={}
    flowa=torch.exp(W['core.flow.a']); sp=F.softplus(W['core.delta_in']); abar=torch.exp(-sp*torch.exp(W['core.a_in']))
    seqs_of=[np.where(seq_si==s)[0] for s in range(S_)]
    for k in range(Kmax):
        act_s=np.where(nb>k)[0]
        pairs_n=[]; pairs_j=[]
        for s in act_s:
            js=R['atoms_sk'][(s,k)]; ns=seqs_of[s]
            pairs_n.append(np.repeat(ns,len(js))); pairs_j.append(np.tile(js,len(ns)))
        pn=np.concatenate(pairs_n); pj=np.concatenate(pairs_j)
        act_n=np.concatenate([seqs_of[s] for s in act_s]); act_n.sort()
        tk=tb[seq_si[act_n],k]
        dt_h=np.where(k==0, 1.0/3600, tk-np.nan_to_num(tb[seq_si[act_n],k-1] if k>0 else tk))
        dt_s=torch.as_tensor(dt_h*3600.0,dtype=torch.float32)
        # ---- heads for meas atoms of bundle k (state after k-1)
        mm=ismeas[pj] & (hidden[pj] | want_nll)
        if k>0 and mm.any():
            hn_=pn[mm]; hj=pj[mm]
            un,inv=np.unique(hn_,return_inverse=True)
            pos=np.searchsorted(act_n,un)
            base=head_base(out[un],h[un],dt_s[pos],W,10.0)
            pr=mlp(base,W,'z_prior'); zmu,zls=pr[:,:32],pr[:,32:]
            eps=EPS[seq_si[un],seq_p[un],k]                     # n x 4 x 32
            c_=torch.as_tensor(code[hj],dtype=torch.long)
            has=reg_has[hn_,code[hj]]; agev=np.where(has,(tt[hj]-reg_t[hn_,code[hj]])*3600.0,0.0)
            zlv=np.where(has,reg_z[hn_,code[hj]],0.0); lbv=np.where(has,reg_b[hn_,code[hj]],0)
            f8=np.zeros((len(hj),8),np.float32); f8[:,0]=has; f8[:,3]=np.log1p(agev)/10; f8[:,6]=zlv
            f8=torch.as_tensor(f8); lb_=torch.as_tensor(lbv,dtype=torch.long); st_=torch.zeros(len(hj),dtype=torch.long)
            P=0
            for d in range(4):
                z=zmu+torch.exp(zls)*eps[:,d]
                lg,_=head_value_logits(base[inv],c_,f8,lb_,st_,W,z[inv],1); P=P+lg.softmax(-1)
            P=P/4
            cen=torch.as_tensor(CENT[code[hj]],dtype=torch.float32)
            pmv=(P[:,1:]*cen).sum(1).numpy()
            for a_,b_,c2 in zip(hn_,hj,pmv): pass
            key=(hn_.astype(np.int64)*10**7+hj)
            pm_rec.update(dict(zip(key.tolist(),pmv.tolist())))
            if want_nll:
                ob=np.where(np.isnan(val[hj]),0,bfact[hj])
                nll_rec.update(dict(zip(key.tolist(),(-np.log(P[np.arange(len(hj)),torch.as_tensor(ob)].numpy())).tolist())))
            hd=hidden[hj]
            if hd.any():
                Pv=P[hd][:,1:].numpy().astype(np.float64); Pv=Pv/Pv.sum(1,keepdims=True)
                u=UNI[seq_si[hn_[hd]],seq_p[hn_[hd]],A['loc'].values[hj[hd]]]
                b=1+np.minimum((Pv.cumsum(1)<u[:,None]).sum(1),15)
                drawn[hn_[hd],A['loc'].values[hj[hd]]]=b
        # ---- embeddings of bundle k atoms
        xh=X[seq_arm[pn],pj]
        hdp=hidden[pj]
        if hdp.any():
            bsel=drawn[pn[hdp],A['loc'].values[pj[hdp]]]
            xh[torch.as_tensor(hdp)]=Xnv[pj[hdp]]+vtab[code[pj[hdp]],bsel]
        h1=mlp(xh,W,'encode.mlp1') if True else None
        isa=isact[pj]
        pos_p=np.searchsorted(act_n,pn)
        na=len(act_n)
        def pool(maskp,pre):
            hh_=mlp(xh[torch.as_tensor(maskp)],W,pre+'.mlp1')
            s_=torch.zeros(na,192).index_add_(0,torch.as_tensor(pos_p[maskp]),hh_)
            cnt=torch.zeros(na).index_add_(0,torch.as_tensor(pos_p[maskp]),torch.ones(int(maskp.sum())))
            pooled=s_/torch.sqrt(cnt.clamp(min=1))[:,None]+F.linear(torch.log1p(cnt)[:,None],W[pre+'.size.weight'],W[pre+'.size.bias'])
            return LN(mlp(pooled,W,pre+'.mlp2'),W,pre+'.norm')
        e=pool(~isa,'encode'); u=pool(isa,'encode_action')
        # ---- core step
        hp=h[act_n]; op=out[act_n]
        hf=hp*torch.exp(-dt_s[:,None,None]*flowa)
        ctx=op+LN(hf.mean(1),W,'core.out_norm')+mlp(torch.log(dt_s)[:,None]/10,W,'gap')
        z=mlp(torch.cat([ctx,e,u],-1),W,'z_post')[:,:32]
        Xt=torch.cat([hf+W['core.slots'][None],e[:,None],u[:,None],F.linear(z,W['z_in.weight'],W['z_in.bias'])[:,None]],1)
        Xt=layer_sa(Xt,W,'core.prelude.0'); Ep=LN(Xt[:,:4],W,'core.prelude_norm')
        ages=np.where(np.arange(Kmax)[None,:k+1]<=k,(tk[:,None]-tb[seq_si[act_n],:k+1])*3600.0,0.0)
        agee=mlp(torch.log1p(torch.as_tensor(ages,dtype=torch.float32))[...,None],W,'core.age')
        x=hf+Ep
        for l in L:
            Wkv=W[l+'.cross.kv.weight']
            Mk[l][act_n,k]=F.linear(e,Wkv[:192]); Mv[l][act_n,k]=F.linear(e,Wkv[192:])
            xn=LN(x,W,l+'.norm1'); x=x+attn_(xn,xn,W,l+'.attn')
            xn=LN(x,W,l+'.norm2')
            kk_=Mk[l][act_n,:k+1]+F.linear(agee,Wkv[:192]); vv_=Mv[l][act_n,:k+1]+F.linear(agee,Wkv[192:])
            x=x+attn_kv(xn,kk_,vv_,W,l+'.cross',None)
            x=x+mlp(LN(x,W,l+'.norm3'),W,l+'.ffn')
        hn=abar*hf+sp*F.linear(Ep,W['core.b_in.weight'])+(x-(hf+Ep))
        y=layer_sa(F.linear(hn,W['core.c_out.weight']),W,'core.coda.0')
        h[act_n]=y; out[act_n]=LN(y.mean(1),W,'core.out_norm')
        # ---- registers
        mp=ismeas[pj]
        if mp.any():
            n_=pn[mp]; j_=pj[mp]; c_=code[j_]
            hdm=hidden[j_]
            bb=np.where(hdm,drawn[n_,A['loc'].values[j_]],bfact[j_])
            zz=np.where(hdm,_z_vec(c_,CENT[c_,np.maximum(bb,1)-1]),zfact[j_])
            ok=~np.isnan(val[j_])|hdm
            reg_has[n_[ok],c_[ok]]=True; reg_t[n_[ok],c_[ok]]=tt[j_[ok]]; reg_z[n_[ok],c_[ok]]=zz[ok]; reg_b[n_[ok],c_[ok]]=bb[ok]
    return dict(pm=pm_rec,nll=nll_rec,seq_si=seq_si,seq_arm=seq_arm,seq_p=seq_p,drawn=drawn)
def _z_vec(c,x):
    med=np.array([NUMD[(CNAME[ci],'value')]['median'] if (CNAME[ci],'value') in NUMD else 0 for ci in range(13)])
    iqr=np.array([NUMD[(CNAME[ci],'value')]['iqr'] if (CNAME[ci],'value') in NUMD else 1 for ci in range(13)])
    return np.clip((x-med[c])/iqr[c],-10,10)
def readout(R,res,arms,drug='insulin',outcome='glucose_finger',t0=12.0,arm_ids=None,lo=13,hi=24):
    A=R['A']; code=A.code.values; tt=A.t.values
    rows=np.where((code==CODE[outcome])&(tt>=lo)&(tt<=hi+1e-9))[0]
    dr=A[(A.code==CODE[drug])&(A.kind==0)&(A.t>=t0)]
    first_dr=dr.groupby('si').t.min()
    reached=np.array([ (A.si.values[j] in first_dr.index) and (first_dr[A.si.values[j]]<tt[j]) for j in rows])
    pm=res['pm']; seq_si=res['seq_si']; seq_arm=res['seq_arm']; seq_p=res['seq_p']
    nP=seq_p.max()+1; nA=seq_arm.max()+1
    out={}
    for a in range(nA):
        vals=np.full((len(rows),nP),np.nan)
        for ri,j in enumerate(rows):
            s=A.si.values[j]
            for p in range(nP):
                n=(s*nA+a)*nP+p
                vals[ri,p]=pm.get(n*10**7+j,np.nan)
        out[a]=vals
    return rows,reached,out
'''
open('/work/rollout.py','w').write(code_roll); exec(code_roll)
sel1_=json.load(open('/work/sel1.json')) if os.path.exists('/work/sel1.json') else sel1
ids30=sorted(sel1)[:30]
R30=prep_rollout(tu,ids30)
arms5=[{},{'ins':2.0},{'ins':0.5},{'ne':2.0},{'ne':0.5}]
import time; t0=time.time()
res30=run_rollout(R30,W,arms5,n_paths=20,seed=0)
print('time',time.time()-t0)
for drug,outc,aa in [('insulin','glucose_finger',[1,2]),('norepinephrine','map_arterial',[3,4])]:
    rows,reached,o=readout(R30,res30,arms5,drug,outc)
    for a in aa:
        sh=np.nanmean(o[a]-o[0],1)
        print(drug,a,'mine',round(np.nanmean(sh[reached]),3),'n',reached.sum())
    # compare with tool
    A_=R30['A']; st_=R30['stays'][A_.si.values[rows]]; hr_=np.round(A_.t.values[rows]).astype(int)
    tool=de1[de1.stay_id.isin(ids30)&(de1.edit==('insulin_x2' if drug=='insulin' else 'ne_x2'))]
    mine=pd.DataFrame({'stay_id':st_,'hour':hr_,'mine_fact':np.nanmean(o[0],1),'mine_sh':np.nanmean(o[aa[0]]-o[0],1),'reached':reached})
    mg=tool.merge(mine,on=['stay_id','hour'],how='inner',suffixes=('','_m'))
    print(' tool rows',len(tool),'merged',len(mg),'reached agree',(mg.reached==mg.reached_m).mean(),
          'tool mean shift(reached)',round(mg[mg.reached].shift.mean(),3),'mine',round(mg[mg.reached].mine_sh.mean(),3),
          'corr fact',round(np.corrcoef(mg.factual,mg.mine_fact)[0,1],4),'rmse fact',round(np.sqrt(np.mean((mg.factual-mg.mine_fact)**2)),2))
```