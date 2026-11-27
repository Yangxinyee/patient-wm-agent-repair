The architecture text changes the picture in two ways.

1. **What the value head reads.** The glucose and MAP heads use the "view": `out` + the out-norm of the time-flowed slot mean + a gap embedding + z_head(z drawn from the prior). They also have a direct register of the last observed value of the same code (its z-score and bins). That register is why the model's prediction leans so heavily on the last reading, and why drawn readings carry a shift forward in a rollout.
2. **The dose has one entry point.** The rate enters only through the action token in the prelude, then E → new slots → `out` and the flowed slots. My earlier estimate counted only `out`. The view also contains the layer-normed mean of the decayed new slots, which change about 200 times more per unit of injection. The fast-decaying dimensions also shrink before that norm, so this term may be much more sensitive.

Next I'll compute the full sensitivity on the same 120 drug bundles:
- the Jacobian of both the new slots and `out` with respect to an injection after prelude attention;
- the exact context for the following bundle, using the real gap and the per-dimension decay `flow.a`;
- a ridge readout from that context to the original model's predicted glucose and MAP means.

From these I'll get the injection norm needed per mg/dL and per mmHg.
- **Prelude carrier viable:** a much smaller norm than the 23 and 34 obtained through `out` alone.
- **Effectively capped:** the requirement stays in the tens, so the effect is limited by the model's architecture.

```python
fa_=fa.copy()
fl_a=load_original('core.flow.a').double(); onw=load_original('core.out_norm.weight').double(); onb=load_original('core.out_norm.bias').double()
def ctx_np(OUTp, Hp, dts):  # OUTp (n,192) Hp (n,4,192) dts seconds (n,)
    O=torch.as_tensor(OUTp,dtype=torch.float64); H=torch.as_tensor(Hp,dtype=torch.float64); dt=torch.as_tensor(dts,dtype=torch.float64)[:,None,None]
    hd=H*torch.exp(-dt*torch.exp(fl_a))[...]
    m=hd.mean(1); return O+LNd(m,onw,onb)
# ridge on ctx (without gap) + log dt features, original model
Co2=ctx_np(xo[ip,:192], xo[ip,192:960].reshape(-1,4,192), dts).numpy()
Fe=np.c_[Co2, np.log(dts), np.log(dts)**2]
WRc={}
for ev in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    m=evs==ev; r=RidgeCV(alphas=np.logspace(-4,2,13)).fit(Fe[m],pm[m]); pr=r.predict(Fe[m])
    WRc[ev]=r; print(f'{ev}: ctx-ridge R2 {1-((pm[m]-pr)**2).mean()/pm[m].var():.5f} alpha {r.alpha_:.3g}')
with torch.no_grad():
    k_=di[sel2]; nxt_ok=np.array([(k+1<len(sid)) and sid[k+1]==sid[k] for k in k_]); dtn=np.array([(hr[k+1]-hr[k])*3600 if ok else 3600. for k,ok in zip(k_,nxt_ok)])
    dtt=torch.as_tensor(dtn,dtype=torch.float64)[:,None,None]; dec=torch.exp(-dtt*torch.exp(fl_a))
    Jc=np.zeros((len(sel2),192,192)); h=2.0
    for j in range(192):
        dx=torch.zeros(len(sel2),4,192,dtype=torch.float64); dx[:,:,j]=h
        res=[]
        for sgn in (1,-1):
            Ep=prelude_tail(Tb,sgn*dx); y_,o_=step_full(k_,Ep.float(),htft[sel2],pre=True)
            c_=o_.double()+LNd((y_.double()*dec).mean(1),onw,onb); res.append(c_)
        Jc[:,:,j]=np.asarray(res[0]-res[1])/(2*h)
for ev,lab in [('glucose_finger','mg/dL'),('map_arterial','mmHg'),('glucose_lab','mg/dL'),('map_cuff','mmHg')]:
    w=WRc[ev].coef_[:192]; g=np.einsum('o,noj->nj',w,Jc); gm=g.mean(0)
    print(f'{ev}: |grad| mean-dir {np.linalg.norm(gm):.4f} {lab} per unit dx (median per-bundle {np.median(np.linalg.norm(g,axis=1)):.4f}); consistency cos median {np.median(g@gm/np.linalg.norm(g,axis=1)/np.linalg.norm(gm)):.3f}; => dx norm per 1 {lab}: {1/np.linalg.norm(gm):.2f}')
    globals()['g_'+ev]=gm
print('cos(g_finger, g_art) %.3f, cos(g_finger,g_lab) %.3f, cos(g_art,g_cuff) %.3f'%(cs(g_glucose_finger,g_map_arterial),cs(g_glucose_finger,g_glucose_lab),cs(g_map_arterial,g_map_cuff)))
print('next-bundle dt (h): median %.2f'%(np.median(dtn)/3600))
```