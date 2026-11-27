Run 228 shows that z encodes glucose surprise well: a cross-validated R² of 0.70 for the glucose_finger surprise, which has an SD of 15 mg/dL. That gives a usable "glucose surprise" direction g in z-space, and `zpost` returns just the 32-number posterior mean, so it is easy to patch.

Next I will test whether pushing z along g at insulin bundles moves later glucose predictions. In the factual replica I will add a constant surprise equivalent to −10 and to −30 mg/dL at every insulin bundle. Then I will measure how the glucose_finger predictions change, grouped by time since the last insulin bundle.

What I expect if the lever works: a drop of a sizeable fraction of the injected amount in the next reading, decaying over the following hours. That would confirm z as a lever I can drive with a dose-based term. A tiny or saturating response would refute it.

```python
if 'zpost_orig' not in globals(): zpost_orig=zpost
gz_t=torch.tensor(gz,dtype=torch.float32); INS_KEY={}; INJ_Z=0.0
def bvd_inj(evs):
    e,u=bvd_orig(evs)
    ds=[a[2].dose for a in evs if a[0]=='insulin' and a[1]=='start']
    if ds: INS_KEY[round(float(u.sum()),5)]=float(np.nansum(ds))
    return e,u
def zpost_inj(o,h,dt,e,u,unit=1.0):
    z=zpost_orig(o,h,dt,e,u,unit)
    if INJ_Z and round(float(u.sum()),5) in INS_KEY: z=z+INJ_Z*gz_t
    return z
bvd=bvd_inj; zpost=zpost_inj
insmin={s:[bm for bm,at in BUN[s] if any(a[0]=='insulin' and a[1]=='start' for a in at)] for s in stays100}
def lagcls(s,mi):
    p=[x for x in insmin[s] if x<mi]; return 9 if not p else min(int((mi-p[-1])//60),4)
base=None
for iz in [0.0,-10.0,-30.0]:
    INJ_Z=iz; CACHE.clear(); o=amp_stats(1.0,stays100)
    if base is None: base=o
    d=np.array([x[3]-y[3] for x,y in zip(o,base)]); ev=np.array([x[1] for x in o]); lc=np.array([lagcls(x[0],x[2]) for x in o])
    nl=np.mean([-np.log(max(x[5][fb_(x[1],x[6])],1e-12)) for x in o])
    print(iz,'nll %.4f'%nl,' '.join('L%d:%.2f(n%d)'%(L_,d[(ev=='glucose_finger')&(lc==L_)].mean(),((ev=='glucose_finger')&(lc==L_)).sum()) for L_ in [0,1,2,3,4]),'MA %.2f'%d[ev=='map_arterial'].mean())
INJ_Z=0.0; CACHE.clear()
```