The `b_in` route is far too weak. Producing the belief shift of a 10 mg/dL reading change through `sp·b_in` would need |ΔE| ≈ 152, ten times the norm of E itself (14). The belief direction lies in `b_in`'s weak subspace (singular values from 0.001 to 5.6, median 0.46).

So I cannot write the reading-like belief shift through E. A different direction might still do the job, because:
- in the rollout, sampled readings carry a shift forward through the model's own belief updates;
- the dose is applied again every hour.

What matters is how strongly the next predicted glucose (MAP) responds to a change in E, through all paths: `b_in`, the blocks and the coda.

Next, at 30 drug bundles, I will measure by finite differences the gradient of the next predicted glucose and MAP means with respect to an E perturbation shared by the four slots. The size of the averaged gradient, and how consistent it is across bundles, tell me whether a modest gated change in E (|ΔE| of a few units) can give shifts of 2–8 mg/dL.

```python
_cs_old=cstep_h
def cstep_h(out,h,dt_s,e,u,Mv,Mage,inj=None,rec=None,dE=None):
    hf=h*torch.exp(-dt_s*torch.exp(W64['core.flow.a']))
    ctx=out+ln_(hf.mean(0),'core.out_norm')+mlp_(torch.tensor([math.log(max(dt_s,1e-9))/10],dtype=torch.float64),'gap')
    zz=mlp_(torch.cat([ctx,e,u]),'z_post')[:32]
    Tk=torch.cat([hf+W64['core.slots'],e[None],u[None],lin_(zz,'z_in')[None]])
    n1=ln_(Tk,'core.prelude.0.norm1'); Tk=Tk+att_(n1,n1,'core.prelude.0.attn')
    ph=Fg(lin_(ln_(Tk,'core.prelude.0.norm3'),'core.prelude.0.ffn.0')); Tk=Tk+lin_(ph,'core.prelude.0.ffn.2')
    E=ln_(Tk[:4],'core.prelude_norm')
    if dE is not None: E=E+dE
    xb=hf+E
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
sel=[d for d in D if isdrug(d) and d['sid'] in FIT and d['k']+1<len(bundles_B[d['sid']]) and (d['sid'],d['k']+1,'glucose_finger') in idx and (d['sid'],d['k']+1,'map_arterial') in idx][::20][:30]
GE=[];eps=0.5
for d in sel:
    s,k=d['sid'],d['k']; BB=bundles_B[s]; r=RK[s][k]; e,u=enc2(BB[k][1]); dt=(BB[k+1][0]-BB[k][0])*3600.0
    ig,im=idx[(s,k+1,'glucose_finger')],idx[(s,k+1,'map_arterial')]
    o0,y0=cstep_h(r['out0'],r['h0'],r['dt'],e,u,r['Mv'],r['Ma'])[:2]; b=(mhead(o0,y0,dt,ig),mhead(o0,y0,dt,im))
    g=np.zeros((2,192))
    for j in range(192):
        dE_=torch.zeros(4,192,dtype=torch.float64); dE_[:,j]=eps
        o1,y1=cstep_h(r['out0'],r['h0'],r['dt'],e,u,r['Mv'],r['Ma'],dE=dE_)[:2]
        g[:,j]=[(mhead(o1,y1,dt,ig)-b[0])/eps,(mhead(o1,y1,dt,im)-b[1])/eps]
    GE.append(g)
GE=np.array(GE); gm=GE.mean(0)
cosb=[np.dot(GE[i,0],gm[0])/np.linalg.norm(GE[i,0])/np.linalg.norm(gm[0]) for i in range(len(GE))]
print('|mean grad| glc %.3f map %.3f | mean |grad| glc %.3f map %.3f | cos to mean glc q10/50 %.2f %.2f'%(np.linalg.norm(gm[0]),np.linalg.norm(gm[1]),np.linalg.norm(GE[:,0],axis=1).mean(),np.linalg.norm(GE[:,1],axis=1).mean(),np.quantile(cosb,.1),np.median(cosb)))
```