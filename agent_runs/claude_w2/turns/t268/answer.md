The four mlp1 unit designs are exact. The fit residual is ~0, the units sit in GELU's linear regime when on (pre-activation ≥ 4.0) and are cleanly off for the other drug (≤ −5). So per START atom the pair difference equals the drug's rate, with no contribution from the rate-bin embeddings.

The pooled vectors have many exactly zero-variance directions, which gives the channel private room after mlp1. 25% of drug bundles carry both drugs, where pooling divides by √2. I'll undo that with a pair of mlp2 units per drug switched by a set-size gate.

Now I build the whole encoder-plus-prelude channel in one function, saved to /work/chan.py so it survives restarts:
- **mlp1:** 4 repurposed low-activity units whose output columns lie on pooled null directions.
- **mlp2:** the original units project those directions out, with bias compensation. Four new size-gated units write into x2.
- **Into u:** the x2 direction is chosen so that after both LayerNorms the change in the u token lies on a quiet direction d̂ of the prelude-token space.
- **Prelude:** `attn.kv` stops all heads from reading d̂ and gives head 2 a private value dimension v̂ (quiet in head 2's outputs). `attn.out` maps v̂ to the slot direction with the largest linear gain into `out`. z_post's u input also stops reading the channel.

This run builds it and checks the encoder alone. Δu along the channel should be linear in rate (R² ≈ 1), there should be no cross-talk between drugs, and the empty-set u should be unchanged. Departures there would refute the construction.

```python
code_chan = r'''
import numpy as np, torch, torch.nn.functional as F
def _np(t): return (t.detach().numpy() if torch.is_tensor(t) else np.asarray(t)).astype(np.float64)
def _gelu(x): return F.gelu(torch.as_tensor(x)).numpy()
def _ln(X,g,b,eps=1e-5):
    m=X.mean(-1,keepdims=True); v=((X-m)**2).mean(-1,keepdims=True); return (X-m)/np.sqrt(v+eps)*g+b
def _lnstd(X,eps=1e-5):
    m=X.mean(-1,keepdims=True); return np.sqrt(((X-m)**2).mean(-1)+eps)
def _small_dirs(M,C=None,k=2):
    n=M.shape[0]
    B=np.linalg.svd(C.T)[2][C.shape[1]:].T if C is not None else np.eye(n)
    ev,evec=np.linalg.eigh(B.T@M@B); return B@evec[:,:k], ev[:k]
def enc_u(Wn,XU,key):
    W1,b1,W2,b2=[Wn['encode_action.mlp1.'+s] for s in ['0.weight','0.bias','2.weight','2.bias']]
    W3,b3,W4,b4=[Wn['encode_action.mlp2.'+s] for s in ['0.weight','0.bias','2.weight','2.bias']]
    O1=_gelu(XU@W1.T+b1)@W2.T+b2
    uk,inv=np.unique(key,return_inverse=True); S=np.zeros((len(uk),O1.shape[1])); np.add.at(S,inv,O1); cnt=np.bincount(inv).astype(float)
    pe=Wn['encode_action.size.bias'][None]
    pool=S/np.sqrt(cnt)[:,None]+np.log1p(cnt)[:,None]*Wn['encode_action.size.weight'][:,0]+pe
    m2=lambda p: _gelu(p@W3.T+b3)@W4.T+b4
    x2=m2(pool); xe=m2(pe); g,b=Wn['encode_action.norm.weight'],Wn['encode_action.norm.bias']
    return dict(u=_ln(x2,g,b),ue=_ln(xe,g,b)[0],pool=pool,pe=pe[0],x2=x2,xe=xe[0],cnt=cnt,inv=inv)
def design_build(W,Dd,P):
    Wn={k:_np(v).copy() for k,v in W.items()}; O={k:v.copy() for k,v in Wn.items()}; rep={}
    XU,isIu,key,SOL=Dd['XU'],Dd['isIu'],Dd['key'],Dd['SOL']
    E0=enc_u(O,XU,key); pool0,pe,cnt0=E0['pool'],E0['pe'],E0['cnt']
    # pooled channel dirs: centered cov (empty weighted 1/2)
    Pd=np.vstack([pool0,np.repeat(pe[None],len(pool0)//2,0)]); mu=Pd.mean(0); C=np.cov(Pd.T)
    ev,evec=np.linalg.eigh(C); c1I,c1N=evec[:,0],evec[:,1]; rep['ev_pool']=ev[:2]
    # mlp1 units
    u1=P['u1']; W1,b1,W2,b2=[Wn['encode_action.mlp1.'+s] for s in ['0.weight','0.bias','2.weight','2.bias']]
    act1=_gelu(XU@O['encode_action.mlp1.0.weight'].T+O['encode_action.mlp1.0.bias'])
    for nm,u,c,sg in zip(['I1','I2','N1','N2'],u1,[c1I,c1I,c1N,c1N],[1,-1,1,-1]):
        b2+=O['encode_action.mlp1.2.weight'][:,u]*act1[:,u].mean()
        W1[u]=SOL[nm][:192]; b1[u]=SOL[nm][192]; W2[:,u]=sg*P['s1']*c
    # mlp2: project out c1 dirs from orig units, gated channel units
    W3,b3,W4,b4=[Wn['encode_action.mlp2.'+s] for s in ['0.weight','0.bias','2.weight','2.bias']]
    for c in [c1I,c1N]:
        wc=W3@c; b3+=wc*(c@mu); W3-=np.outer(wc,c)
    nd=np.r_[cnt0,np.zeros(1)]; Pg=np.vstack([pool0,pe[None]])
    Pp=Pg-np.outer(Pg@c1I,c1I)-np.outer(Pg@c1N,c1N)
    sol=np.linalg.lstsq(np.c_[Pp,np.ones(len(Pp))],np.where(nd<=1,1.0,-1.0),rcond=None)[0]
    g=sol[:192]-c1I*(c1I@sol[:192])-c1N*(c1N@sol[:192]); g0=sol[192]; gp=Pg@g+g0
    rep['gate_err']=np.abs(gp-np.where(nd<=1,1.0,-1.0)).max()
    act2=_gelu(pe@O['encode_action.mlp2.0.weight'].T+O['encode_action.mlp2.0.bias'])
    for u in P['u2']: b4+=O['encode_action.mlp2.2.weight'][:,u]*act2[u]
    B,G,beta=P['B'],P['G'],P['beta']; mI,mN=c1I@mu,c1N@mu
    # target directions: dhat in LN1 space (quiet over all prelude tokens), mean-constraints
    xn=Dd['xn']; g1=Wn['core.prelude.0.norm1.weight']; ge=Wn['encode_action.norm.weight']
    Mx=np.einsum('ntd,nte->de',xn,xn)/(xn.shape[0]*xn.shape[1])
    Cc=np.c_[1/g1,1/(g1*ge)]; Dh,evx=_small_dirs(Mx,Cc,2); rep['ev_xn']=evx
    su=_lnstd(Dd['X7'][:,5]).mean(); sx2=_lnstd(E0['x2']).mean(); rep['su,sx2']=(su,sx2)
    c2={}
    for i,(dn,kap) in enumerate([('I',P['kapI']),('N',P['kapN'])]):
        c2[dn]=sx2*su*kap/(beta*P['s1'])*(Dh[:,i]/(g1*ge))
    for (uA,uB),c,m,dn in [((P['u2'][0],P['u2'][1]),c1I,mI,'I'),((P['u2'][2],P['u2'][3]),c1N,mN,'N')]:
        W3[uA]=beta*c+G*g; b3[uA]=B-beta*m+G*g0-G
        W3[uB]=np.sqrt(2)*beta*c-G*g; b3[uB]=B-np.sqrt(2)*beta*m-G*g0-G
        W4[:,uA]=c2[dn]; W4[:,uB]=c2[dn]; b4-=c2[dn]*B
    # prelude kv: remove dhat reading, head-2 private value dims
    K=Wn['core.prelude.0.attn.kv.weight']; att=Dd['att']
    vt=np.einsum('ntd,ed->nte',xn,O['core.prelude.0.attn.kv.weight'][192+96:192+144])
    oh=np.einsum('nst,nte->nse',att[:,2,:4,:],vt); Mv=np.einsum('nse,nsf->ef',oh,oh)/(oh.shape[0]*4)
    evv,evecv=np.linalg.eigh(Mv); vh=evecv[:,:2]; rep['ev_v']=evv[:2]
    for i in range(2):
        d=Dh[:,i]; K-=np.outer(K@d,d)
    for i in range(2): K[192+96:192+144]+=P['gv']*np.outer(vh[:,i],Dh[:,i])
    # out projection target dirs: max linear gain into out
    sp=np.log1p(np.exp(Wn['core.delta_in'])); sp=sp.reshape(-1,192).mean(0) if sp.size>=192 else np.full(192,float(sp))
    gpn=Wn['core.prelude_norm.weight']; gon=Wn['core.out_norm.weight']
    sX=_lnstd(Dd['X7'][:,:4]).mean(); sy=Dd['sy']; Pc=np.eye(192)-1/192
    J=np.diag(gon/sy)@Pc@Wn['core.c_out.weight']@np.diag(sp)@Wn['core.b_in.weight']@np.diag(gpn/sX)@Pc
    Uj,Sj,Vj=np.linalg.svd(J); c3=Vj[:2].T; rep['sJ']=Sj[:2]
    Wo=Wn['core.prelude.0.attn.out.weight']; blk=Wo[:,96:144]
    blk=blk@(np.eye(48)-vh@vh.T)
    for i,(dn,kap,tgt) in enumerate([('I',P['kapI'],P['tgtI']),('N',P['kapN'],P['tgtN'])]):
        g3=tgt/(Sj[i]*P['gv']*kap); blk+=g3*np.outer(c3[:,i],vh[:,i])
    Wo[:,96:144]=blk
    # z_post: stop reading channel directions of u
    zp=[n for n in Wn if n.startswith('z_post') and n.endswith('0.weight')][0]; zb=zp.replace('weight','bias')
    Zw=Wn[zp]; ub=Dd['X7'][:,5].mean(0)
    for i in range(2):
        du=Dh[:,i]/g1; du=du/np.linalg.norm(du); wz=Zw[:,384:576]@du; Wn[zb]+=wz*(du@ub); Zw[:,384:576]-=np.outer(wz,du)
    rep.update(dict(c1I=c1I,c1N=c1N,Dh=Dh,vh=vh,c3=c3,g=g,g0=g0))
    return {k:torch.as_tensor(v,dtype=torch.float32) for k,v in Wn.items()}, rep
'''
open('/work/chan.py','w').write(code_chan); exec(code_chan)
SOL['N2']=solve_unit(5.04+0*zN, 5.04-10+0*zI, ~isIu)[0]
xn_=xn.numpy() if torch.is_tensor(xn) else xn; X7_=X7.numpy() if torch.is_tensor(X7) else X7
sy_=_lnstd(Cs['y'].reshape(-1,4,192)).mean()
Dd=dict(XU=XU,isIu=isIu,key=key,SOL=SOL,xn=xn_.astype(np.float64),X7=X7_.astype(np.float64),att=at.astype(np.float64),sy=sy_)
P=dict(u1=[160,447,722,230],u2=[468,219,217,433],s1=1.0,beta=1.0,B=4.0,G=12.0,kapI=0.3,kapN=1.0,gv=1.0,tgtI=0.3,tgtN=1.0)
print('xn check', float(np.abs(_ln(X7_[:50].astype(np.float64),_np(W['core.prelude.0.norm1.weight']),_np(W['core.prelude.0.norm1.bias']))-xn_[:50]).max()))
Wc,rep=design_build(W,Dd,P)
print({k:(np.round(v,5) if isinstance(v,np.ndarray) and v.size<5 else v) for k,v in rep.items() if k not in ['c1I','c1N','Dh','vh','c3','g']})
E0=enc_u({k:_np(v) for k,v in W.items()},XU,key); E1=enc_u({k:_np(v) for k,v in Wc.items()},XU,key)
du=E1['u']-E0['u']; inv_=E0['inv']
isIb=np.zeros(len(du),bool); np.logical_or.at(isIb,inv_,isIu); isNb=np.zeros(len(du),bool); np.logical_or.at(isNb,inv_,~isIu)
rIb=np.zeros(len(du)); np.add.at(rIb,inv_,np.where(isIu,RU,0)); rNb=np.zeros(len(du)); np.add.at(rNb,inv_,np.where(~isIu,RU,0))
g1_=_np(W['core.prelude.0.norm1.weight'])
for i,(nm_,rr) in enumerate([('I',rIb),('N',rNb)]):
    dd=rep['Dh'][:,i]/g1_; dd=dd-dd.mean(); proj=du@dd/(dd@dd)
    lr_=np.polyfit(rr,proj,1); r2_=1-np.var(proj-np.polyval(lr_,rr))/np.var(proj)
    print(nm_,'slope %.4f int %.4f R2 %.5f'%(lr_[0],lr_[1],r2_), 'cnt2 slope %.4f'%np.polyfit(rr[E0['cnt']==2],proj[E0['cnt']==2],1)[0])
print('|du| resid (orth to chans) mean %.4f, |u| mean %.2f, empty change %.5f'%(np.linalg.norm(du-np.outer(du@rep['Dh'][:,0]/g1_,np.zeros(192)),axis=1).mean(),np.linalg.norm(E0['u'],axis=1).mean(),np.abs(E1['ue']-E0['ue']).max()))
```