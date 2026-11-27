The gate cannot be built where I planned. On held-out stays the dose target is not linearly decodable from the last block FFN's hidden units (R² about 0 or negative at every penalty).

**Why:** inside the blocks the residual stream is dominated by the very large recurrent state (|h| ≈ 1,800). The prelude output E, which does carry the dose (about 7.5% change under x2), is only a small addition there, so the dose signal is swamped.

**Where the dose does reach h_new:** linearly, through `sp·b_in(E)`.

**New design:** read the gate from E, and write through a low-rank update of `core.b_in.weight`. Then Δh_new = sp ⊙ a·(q·E) exactly, with no other effect within the step.

Next I will re-collect E, and the prelude FFN hidden units for comparison, at every bundle and every dose variant. Then I will repeat the held-out ridge test. Good R² with low leakage from E would confirm the `b_in` route.

```python
def cstep_h(out,h,dt_s,e,u,Mv,Mage,inj=None,rec=None):
    hf=h*torch.exp(-dt_s*torch.exp(W64['core.flow.a']))
    ctx=out+ln_(hf.mean(0),'core.out_norm')+mlp_(torch.tensor([math.log(max(dt_s,1e-9))/10],dtype=torch.float64),'gap')
    zz=mlp_(torch.cat([ctx,e,u]),'z_post')[:32]
    Tk=torch.cat([hf+W64['core.slots'],e[None],u[None],lin_(zz,'z_in')[None]])
    n1=ln_(Tk,'core.prelude.0.norm1'); Tk=Tk+att_(n1,n1,'core.prelude.0.attn')
    ph=Fg(lin_(ln_(Tk,'core.prelude.0.norm3'),'core.prelude.0.ffn.0')); Tk=Tk+lin_(ph,'core.prelude.0.ffn.2')
    E=ln_(Tk[:4],'core.prelude_norm'); xb=hf+E
    if rec is not None: rec['E']=E; rec['ph']=ph[:4]
    Mem=Mv+mlp_(torch.log1p(Mage)[:,None],'core.age')
    for l in range(2):
        p='core.blocks.0.%d'%l
        a=ln_(xb,p+'.norm1'); xb=xb+att_(a,a,p+'.attn')
        xb=xb+att_(ln_(xb,p+'.norm2'),Mem,p+'.cross')
        xb=xb+mlp_(ln_(xb,p+'.norm3'),p+'.ffn')
    sp=Fn.softplus(W64['core.delta_in']); abar=torch.exp(-sp*torch.exp(W64['core.a_in']))
    hn=abar*hf+sp*lin_(E,'core.b_in')+(xb-(hf+E))
    if inj is not None: hn=hn+inj
    x=hn@W64['core.c_out.weight'].T
    a=ln_(x,'core.coda.0.norm1'); x2=x+att_(a,a,'core.coda.0.attn')
    y=x2+mlp_(ln_(x2,'core.coda.0.norm3'),'core.coda.0.ffn')
    return ln_(y.mean(0),'core.out_norm'),y,zz,E,hf,hn
t0=time.time(); VN=dict(i2=('insulin',2.),ih=('insulin',.5),n2=('norepinephrine',2.),nh=('norepinephrine',.5))
for d in D:
    s,k=d['sid'],d['k']; BB=bundles_B[s]; r=RK[s][k]; at=BB[k][1]
    rec=substep(RK[s],BB,k,r['out0'],r['h0'],at)[1]; d['E']=rec['E'].float(); d['ph']=rec['ph'].float()
    if 'hid_i2' in d:
        for nm,(cd,f) in VN.items():
            rc=substep(RK[s],BB,k,r['out0'],r['h0'],scale(at,cd,f))[1]; d['E_'+nm]=rc['E'].float(); d['ph_'+nm]=rc['ph'].float()
print('done %.0fs'%(time.time()-t0))
```