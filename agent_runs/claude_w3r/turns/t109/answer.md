build_bundles turns a stay's events into time-stamped bundles of atom dicts. It also adds an END atom for each infusion at its end time, so a bundle 5 minutes past the hour holds the new infusion and the END atom of the previous one. The second element of obj is the list of bundle times, and it matches the states rows. My notes say z matched best when END atoms go through the measurement encoder. If so, the previous dose sits in e_k and in memory M, and only new infusions go through u_k.

Next I will write a local version of one full core step and test it on stay s0 against the model's real states. The step covers time flow, prelude, two recurrent layers with memory cross-attention, the update, coda and out. I start from the model's own previous state and posterior z, and try variants for the unknown details: age units and where the age term enters the keys, whether memory includes the current bundle, whether memory is normalised.

- **Confirms:** some variant reproduces out and h with small error (well under 0.05 median). Then I have a usable local core for designing the edit.
- **Refutes:** all variants are far off. Then my encoders or the token layout are wrong.

```python
import math, torch, itertools, time
F=torch.nn.functional
Wt={n:torch.tensor(np.asarray(load_original(n)),dtype=torch.float64) for n in NAMES}
mm=M({n:load_original(n) for n in NAMES}); mm.W={k:v.double() for k,v in mm.W.items()}
OPTS=dict(unit={'glucose_finger':2,'glucose_lab':2,'map_arterial':3,'map_cuff':3,'insulin':4,'norepinephrine':5},
          ru={'glucose_finger':0,'glucose_lab':0,'map_arterial':0,'map_cuff':0,'insulin':2,'norepinephrine':3}, end_attrs=True, kq=True)
def LNt(x,name): return F.layer_norm(x,(x.shape[-1],),Wt[name+'.weight'],Wt[name+'.bias'],1e-5)
def mha(xq,xk,xv,pre,kadd=None):
    q=xq@Wt[pre+'.q.weight'].T; Wkv=Wt[pre+'.kv.weight']; k=xk@Wkv[:192].T; v=xv@Wkv[192:].T
    if kadd is not None: k=k+kadd
    Tq,Tk=q.shape[0],k.shape[0]
    qh=q.view(Tq,4,48).transpose(0,1); kh=k.view(Tk,4,48).transpose(0,1); vh=v.view(Tk,4,48).transpose(0,1)
    att=torch.softmax(qh@kh.transpose(1,2)/math.sqrt(48),-1)
    return ((att@vh).transpose(0,1).reshape(Tq,192))@Wt[pre+'.out.weight'].T, att
def ffn(x,pre): return F.gelu(x@Wt[pre+'.0.weight'].T+Wt[pre+'.0.bias'])@Wt[pre+'.2.weight'].T+Wt[pre+'.2.bias']
def mlp_t(x,pre): return F.gelu(x@Wt[pre+'.0.weight'].T+Wt[pre+'.0.bias'])@Wt[pre+'.2.weight'].T+Wt[pre+'.2.bias']
def gap_t(dt): return mlp_t(torch.tensor([[math.log(max(dt,1.0))/10.0]],dtype=torch.float64),'gap')[0]
def age_emb(a, sc): return mlp_t(sc(a)[:,None],'core.age')
def enc_eu(atoms):
    obs=[a for a in atoms if not (a['ek']==2 and a['kind']==0)]; act=[a for a in atoms if a['ek']==2 and a['kind']==0]
    e=mm.encode_set(torch.stack([mm.embed_atom(a,OPTS) for a in obs]),'encode') if obs else torch.zeros(192,dtype=torch.float64)
    u=mm.encode_set(torch.stack([mm.embed_atom(a,OPTS) for a in act]),'encode_action') if act else torch.zeros(192,dtype=torch.float64)
    return e,u
def core_step(hprev,outprev,dt,e,u,zk,Mem,Mage,var):
    decay=torch.exp(-dt*torch.exp(Wt['core.flow.a'])); h=hprev*decay
    X=torch.cat([h+Wt['core.slots'], e[None], u[None], (zk@Wt['z_in.weight'].T+Wt['z_in.bias'])[None]],0)
    a_,_=mha(LNt(X,'core.prelude.0.norm1'),LNt(X,'core.prelude.0.norm1'),LNt(X,'core.prelude.0.norm1'),'core.prelude.0.attn'); X=X+a_
    X=X+ffn(LNt(X,'core.prelude.0.norm3'),'core.prelude.0.ffn'); E=LNt(X[:4],'core.prelude_norm')
    x=h+E
    AE=age_emb(Mage,var['sc']) if Mem is not None else None
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        xn=LNt(x,L+'.norm1'); a_,_=mha(xn,xn,xn,L+'.attn'); x=x+a_
        if Mem is not None:
            xn=LNt(x,L+'.norm2'); Mm=LNt(Mem,L+'.norm2') if var['mnorm'] else Mem
            if var['kf']=='post': a_,_=mha(xn,Mm,Mm,L+'.cross',kadd=AE)
            elif var['kf']=='pre': a_,_=mha(xn,Mm+AE,Mm,L+'.cross')
            else: a_,_=mha(xn,Mm+AE,Mm+AE,L+'.cross')
            x=x+a_
        x=x+ffn(LNt(x,L+'.norm3'),L+'.ffn')
    sp=F.softplus(Wt['core.delta_in']); abar=torch.exp(-sp*torch.exp(Wt['core.a_in']))
    hn=abar*h+sp*(E@Wt['core.b_in.weight'].T)+(x-(h+E))
    y=hn@Wt['core.c_out.weight'].T
    yn=LNt(y,'core.coda.0.norm1'); a_,_=mha(yn,yn,yn,'core.coda.0.attn'); y=y+a_
    y=y+ffn(LNt(y,'core.coda.0.norm3'),'core.coda.0.ffn')
    return y, LNt(y.mean(0),'core.out_norm'), E
def stay_data(s):
    ev=recs[recs.stay_id==s].sort_values('hour'); B=build_bundles(ev,OPTS); rows=np.where(SST==s)[0]
    EU=[enc_eu(al) for _,al in B]; return B,rows,EU
B0,rows0,EU0=stay_data(s0); print('n bundles',len(B0),'rows',len(rows0), 'times match', np.allclose([t for t,_ in B0], SHR[rows0], atol=1e-3))
scs={'s':lambda a:torch.log1p(a),'s10':lambda a:torch.log1p(a)/10,'min':lambda a:torch.log1p(a/60),'h':lambda a:torch.log1p(a/3600)}
res_=[]
t0=time.time()
for scn,kf,mn,incl in itertools.product(scs,['post','pre','both'],[False,True],[False,True]):
    var=dict(sc=scs[scn],kf=kf,mnorm=mn); errs=[]
    for k in range(1,len(B0)):
        r_=rows0[k]; hp=torch.tensor(SX[rows0[k-1],192:960].reshape(4,192),dtype=torch.float64); op=torch.tensor(SX[rows0[k-1],:192],dtype=torch.float64)
        dt=(B0[k][0]-B0[k-1][0])*3600; zk=torch.tensor(SX[r_,960:],dtype=torch.float64)
        idx_=list(range(0,k+1 if incl else k))[-128:]
        Mem=torch.stack([EU0[j][0] for j in idx_]) if idx_ else None
        Mage=torch.tensor([(B0[k][0]-B0[j][0])*3600 for j in idx_],dtype=torch.float64) if idx_ else None
        y,o,_=core_step(hp,op,dt,EU0[k][0],EU0[k][1],zk,Mem,Mage,var)
        errs.append(float((o-torch.tensor(SX[r_,:192],dtype=torch.float64)).abs().median()))
    res_.append((np.median(errs),np.max(errs),scn,kf,mn,incl))
res_.sort(); print('time',round(time.time()-t0,1))
for rr_ in res_[:8]: print(rr_)
print('worst', res_[-1])
```