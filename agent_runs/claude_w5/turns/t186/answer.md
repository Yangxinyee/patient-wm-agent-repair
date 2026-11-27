`view0` is exact (error 9e-16), and each `H` row also carries the code, the last-value features and the observed value. So I can score how any state perturbation moves the next predicted reading.

Next I will build the fitting data on 70 of the 100 stays in `bundles_B`, holding out 30 for validation:
- **Gate features:** at every bundle, the 768 hidden activations of the last block's FFN, per slot. At drug bundles, also under each of the four dose edits.
- **Belief direction:** at each drug bundle, the change in h_new when the h:00 glucose reading is 10 lower, and, separately, when both MAP readings are 5 higher.
- **Head response:** how much the next predicted glucose and MAP means move under that state change alone.

This is plain bookkeeping with the exact local step. Success means it finishes in time and gives a few thousand drug-bundle records.

```python
import time
Fg=Fn.gelu
def cstep_h(out,h,dt_s,e,u,Mv,Mage,inj=None,rec=None):
    hf=h*torch.exp(-dt_s*torch.exp(W64['core.flow.a']))
    ctx=out+ln_(hf.mean(0),'core.out_norm')+mlp_(torch.tensor([math.log(max(dt_s,1e-9))/10],dtype=torch.float64),'gap')
    zz=mlp_(torch.cat([ctx,e,u]),'z_post')[:32]
    Tk=torch.cat([hf+W64['core.slots'],e[None],u[None],lin_(zz,'z_in')[None]])
    n1=ln_(Tk,'core.prelude.0.norm1'); Tk=Tk+att_(n1,n1,'core.prelude.0.attn')
    Tk=Tk+mlp_(ln_(Tk,'core.prelude.0.norm3'),'core.prelude.0.ffn')
    E=ln_(Tk[:4],'core.prelude_norm'); xb=hf+E
    Mem=Mv+mlp_(torch.log1p(Mage)[:,None],'core.age')
    for l in range(2):
        p='core.blocks.0.%d'%l
        a=ln_(xb,p+'.norm1'); xb=xb+att_(a,a,p+'.attn')
        xb=xb+att_(ln_(xb,p+'.norm2'),Mem,p+'.cross')
        hid=Fg(lin_(ln_(xb,p+'.norm3'),p+'.ffn.0')); xb=xb+lin_(hid,p+'.ffn.2')
    if rec is not None: rec['hid']=hid
    sp=Fn.softplus(W64['core.delta_in']); abar=torch.exp(-sp*torch.exp(W64['core.a_in']))
    hn=abar*hf+sp*lin_(E,'core.b_in')+(xb-(hf+E))
    if inj is not None: hn=hn+inj
    x=hn@W64['core.c_out.weight'].T
    a=ln_(x,'core.coda.0.norm1'); x2=x+att_(a,a,'core.coda.0.attn')
    y=x2+mlp_(ln_(x2,'core.coda.0.norm3'),'core.coda.0.ffn')
    return ln_(y.mean(0),'core.out_norm'),y,zz,E,hf,hn
def scale(at,code,f): return [dict(a,rate=a['rate']*f,dose=a['dose']*f) if (a['code']==code and a['kind']==0) else a for a in at]
def modread(at,codes,dv): return [dict(a,value=a['value']+dv) if (a['code'] in codes and a['kind']==0 and 'value' in a) else a for a in at]
def substep(Rk,B,k,out,h,at,Mrep=None):
    r=Rk[k]; e,u=enc2(at); Mv=r['Mv'].clone(); Mv[-1]=e
    if Mrep is not None: j,ej=Mrep; Mv[j-(k+1-len(Mv))]=ej
    rec={}; res=cstep_h(out,h,r['dt'],e,u,Mv,r['Ma'],rec=rec); return res,rec,e
sids=list(bundles_B); FIT=sids[:70]; VAL=sids[70:]
D=[]; t0=time.time()
for s in sids:
    BB=bundles_B[s]; Rk=run_tf(BB,keep=True)
    for k,(t,at) in enumerate(BB):
        r=Rk[k]; res,rec,_=substep(Rk,BB,k,r['out0'],r['h0'],at)
        cs={a['code']:a for a in at if a['kind']==0}
        d=dict(sid=s,k=k,t=t,hid=rec['hid'].float(),ri=cs['insulin']['rate'] if 'insulin' in cs else 0.0,rn=cs['norepinephrine']['rate'] if 'norepinephrine' in cs else 0.0)
        drug=('insulin' in cs) or ('norepinephrine' in cs)
        if drug:
            for nm,(cd,f) in dict(i2=('insulin',2.),ih=('insulin',.5),n2=('norepinephrine',2.),nh=('norepinephrine',.5)).items():
                d['hid_'+nm]=substep(Rk,BB,k,r['out0'],r['h0'],scale(at,cd,f))[1]['hid'].float()
            kp=k-1; tp,atp=BB[kp]; cp={a['code']:a for a in atp if a['kind']==0}
            d['g']=cp['glucose_finger']['value'] if 'glucose_finger' in cp else np.nan
            d['ma']=cp['map_arterial']['value'] if 'map_arterial' in cp else np.nan
            d['mc']=cp['map_cuff']['value'] if 'map_cuff' in cp else np.nan
            rp=Rk[kp]; 
            for nm,codes,dv in [('vg',('glucose_finger',),-10.0),('vm',('map_arterial','map_cuff'),5.0)]:
                if not any(a['code'] in codes for a in atp if a['kind']==0): continue
                resp,_,ep=substep(Rk,BB,kp,rp['out0'],rp['h0'],modread(atp,codes,dv))
                resd,_,_=substep(Rk,BB,k,resp[0],resp[1],at,Mrep=(kp,ep))
                d[nm]=((resd[5]-res[5])/abs(dv)).float(); d[nm+'_y']=((resd[1]-res[1])/abs(dv)).float()
                d[nm+'_st']=(resd[0],resd[1])
            d['st']=(res[0],res[1])
        D.append(d)
print(len(D),'recs',sum(1 for d in D if 'hid_i2' in d),'drug',sum(1 for d in D if 'vg' in d),'vg',sum(1 for d in D if 'vm' in d),'vm','%.0fs'%(time.time()-t0))
```