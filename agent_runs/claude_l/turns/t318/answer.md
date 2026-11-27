Run 290 gave me what I need to build a stronger injection:
- **Prelude attention:** queries come from `attn.q.weight`; keys and values are the two halves of `attn.kv.weight`; there are 4 heads of width 48, no biases, and an output map `attn.out.weight`.
- **Bundle format:** each bundle is `(t, atoms)`, where each atom is a dict with code, value bin `vb` and numeric features `num`, so I can edit reading values and re-encode them with `enc_eu2`.
- **Leftover arrays:** `Dtr`/`Dtu` are hourly data tables, not directions. `DIRS`/`Dmin` are earlier attempts, nearly orthogonal to the dose direction, so I won't reuse them.

The plan follows from run 289: the dose direction moves predicted glucose too little per unit of token movement. The strongest lever the model has is the value-message a reading sends into the state. At reading bundles, heads 1 and 2 carry the reading's value from the e token. At drug bundles, the same heads read the u token and its copy.

So I will build the copy-token direction as a "virtual reading" message:
1. **Target:** the head-1/2 value change that a 1 mg/dL higher glucose reading (or 1 mmHg higher MAP) produces, measured by re-encoding real reading bundles with shifted values.
2. **Fit:** the smallest token direction that reproduces that target at drug bundles, given the copy token's actual attention weights. It also keeps head 0's value unchanged and pays a penalty for moving attention scores.

I'll check the direction's norm and linearity, plug it into `mkW3` (insulin residual → glucose-lowering message, norepinephrine residual → MAP-raising message), and sweep the gains with the offline rollout and teacher-forced NLL.

What would support this approach:
- directions with small norms (a few token units), unlike the 1,000+ of run 289;
- near-linear E responses;
- readouts that finally grow with gain, symmetrically for ×2 and ×½.
Saturation again would mean the copy-token route is capped, and I'd fall back to the best safe gain.

```python
import time, math
tA=time.time()
NM2ID={n:CODE_ID[n] for n in ['glucose_finger','glucose_lab','map_arterial','map_cuff']}; ID2NM={v:k for k,v in NM2ID.items()}
st40=list(BUN.keys())[:40]
rbs=[b for s in st40 for b in BUN[s] if any(a_['code'] in ID2NM for a_ in b[1])]
fa=[a_ for b in rbs[:300] for a_ in b[1] if a_['code']==NM2ID['glucose_finger']]
md_,iq_=NUMV[('glucose_finger','value')]['median'],NUMV[('glucose_finger','value')]['iqr']
mt=[np.mean([fbin_f32('glucose_finger','value',a_['num'][i]*iq_+md_)==a_['vb'] for a_ in fa]) for i in range(6)]
bi=int(np.argmax(mt)); print('reading bundles',len(rbs),'val idx match',np.round(mt,3),'->',bi,'| atom', fa[0])
def shift_atoms(atoms,names,dv):
    o_=[]
    for a_ in atoms:
        nm=ID2NM.get(a_['code'])
        if nm in names:
            md,iq=NUMV[(nm,'value')]['median'],NUMV[(nm,'value')]['iqr']; val=a_['num'][bi]*iq+md+dv
            b_=dict(a_); nu=list(a_['num']); nu[bi]=zsc(nm,'value',val); b_['num']=tuple(nu); b_['vb']=fbin_f32(nm,'value',val); o_.append(b_)
        else: o_.append(a_)
    return o_
Wkv9=W0z['core.prelude.0.attn.kv.weight'].to(T); Wv9=Wkv9[192:]; Wk9=Wkv9[:192]; Wq9=W0z['core.prelude.0.attn.q.weight'].to(T); g19=W0z['core.prelude.0.norm1.weight'].to(T)
vals9=lambda x: bln(W0z,x,'core.prelude.0.norm1')@Wv9.T
MSG={}
for grp,names,dv in [('glu',('glucose_finger','glucose_lab'),10.0),('map',('map_arterial','map_cuff'),5.0)]:
    sel=[b for b in rbs if any(ID2NM.get(a_['code']) in names for a_ in b[1])][:500]
    e0=torch.stack([td9(enc_eu2(b[1])[0]).reshape(-1) for b in sel]); e1=torch.stack([td9(enc_eu2(shift_atoms(b[1],names,dv))[0]).reshape(-1) for b in sel])
    dvv=(vals9(e1)-vals9(e0))/dv; MSG[grp]=dvv.mean(0)
    print(grp,len(sel),'|de|/unit %.4f'%float((e1-e0).norm(dim=1).mean()/dv),'msg/head',[round(float(MSG[grp][h*48:(h+1)*48].norm()),4) for h in range(4)],'coh %.3f'%float(F.cosine_similarity(dvv,MSG[grp][None]).mean()))
def lnjac(x,g):
    xc=x-x.mean(-1,keepdim=True); sd=torch.sqrt((xc**2).mean(-1,keepdim=True)+1e-5); xh=xc/sd; D_=x.shape[-1]
    return g[None,:,None]*(torch.eye(D_,dtype=T)[None]-1.0/D_-xh[:,:,None]*xh[:,None,:]/D_)/sd[:,:,None]
def att_w(hf,e,u,z):
    tok=torch.cat([hf+W0z['core.slots'].to(T),e[:,None],u[:,None],z[:,None]],1); a=bln(W0z,tok,'core.prelude.0.norm1'); B=tok.shape[0]
    q=(a[:,:4]@Wq9.T).reshape(B,4,4,48).transpose(1,2); k=(a@Wk9.T).reshape(B,7,4,48).transpose(1,2)
    return torch.softmax(q@k.transpose(-1,-2)/math.sqrt(48),-1)
VDV={}
for d9,grp,sgn in [('ins','glu',-1.0),('ne','map',1.0)]:
    cl9=grp9[d9]; nb9=len(cl9)
    hf9=torch.stack([td9(q['hf']) for q in cl9]); e9=torch.stack([td9(q['e']) for q in cl9]); u9=torch.stack([td9(q['u']) for q in cl9]); x9=torch.stack([td9(q['ctx']) for q in cl9])
    z09=ztok(Wc9,x9,e9,u9); az=att_w(hf9,e9,u9,z09)[...,6]            # (nb,4h,4s)
    Vb=Wv9[None]@lnjac(z09,g19)                                       # (nb,192,192)
    tgt=sgn*MSG[grp].clone(); tgt[0:48]=0
    AtA=torch.zeros(192,192,dtype=T); Atb=torch.zeros(192,dtype=T); sst=0.0
    for h in range(3):
        Vh=Vb[:,h*48:(h+1)*48]; w2=(az[:,h]**2).sum(1); w1=az[:,h].sum(1)
        AtA+=torch.einsum('b,bij,bik->jk',w2,Vh,Vh); Atb+=torch.einsum('b,bij,i->j',w1,Vh,tgt[h*48:(h+1)*48]); sst+=4*nb9*float((tgt[h*48:(h+1)*48]**2).sum())
    KG=torch.stack([td9(keygrads(q)[1]) for q in cl9]); KtK=torch.einsum('bij,bik->jk',KG,KG)
    print(d9,'mean a_z heads0-3', np.round(az.mean((0,2)).numpy(),3))
    for mu0 in [1.0,10.0,100.0]:
        M_=AtA+mu0*float(torch.trace(AtA)/torch.trace(KtK))*KtK+1e-6*float(torch.trace(AtA))/192*torch.eye(192,dtype=T)
        dl=torch.linalg.solve(M_,Atb); res=float(dl@AtA@dl-2*dl@Atb)+sst
        print('  mu0',mu0,'|d| %.4f'%float(dl.norm()),'valR2 %.3f'%(1-res/sst),'|dlogit| %.4f'%float((KG@dl).abs().mean()))
        if mu0==10.0: VDV[d9]=dl.numpy().copy()
    dl=torch.tensor(VDV[d9]); idx9=torch.arange(min(nb9,80)); Ez=E_of(Wc9,hf9[idx9],0.0,e9[idx9],u9[idx9],z09[idx9])
    dE1=(E_of(Wc9,hf9[idx9],0.0,e9[idx9],u9[idx9],z09[idx9]+0.01*dl[None])-Ez)/0.01
    print('  linearity (A, relerr):',[(A,round(float((((E_of(Wc9,hf9[idx9],0.0,e9[idx9],u9[idx9],z09[idx9]+A*dl[None])-Ez)/A-dE1)**2).sum()/(dE1**2).sum()),3)) for A in [5.0,20.0,50.0]])
VD_.update(VDV)
for gi,gn in [(3.0,8.0),(6.0,16.0),(12.0,32.0)]:
    Wn9=mkW3(gi,gn,W0z); rr9=evalW(Wn9,st40); print('MSG gains',gi,gn,{kk:vv[0] for kk,vv in rr9.items()},'tfnll %.5f'%tfnll(Wn9,st40)[0])
print('%.0fs'%(time.time()-tA))
```