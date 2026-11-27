Run 223 confirms that `run_cf` encodes bundles through the global `bvd`, cached via `CACHE`. Patching `bvd` and clearing the cache will therefore change the action encoding u everywhere in the replica.

Next I will run the amplification experiment. At each insulin bundle, u becomes u + s·(u(2d) − u(d)), for s = 0, 2 and 5. For each s I will report the factual NLL, the teacher-forced insulin x2 glucose shift, and how much the factual glucose predictions move. The idea is supported if the shift scales roughly with (1+s). It is refuted if the shift stays near −0.9.

```python
if 'bvd_orig' not in globals(): bvd_orig=bvd
INJ_S=0.0
def bvd_inj(evs):
    e,u=bvd_orig(evs)
    if INJ_S and any(a[0]=='insulin' for a in evs):
        ev2=[(a[0],a[1],a[2]._replace(rate=a[2].rate*2,dose=a[2].dose*2)) if a[0]=='insulin' else a for a in evs]
        u=u+INJ_S*(bvd_orig(ev2)[1]-u)
    return e,u
bvd=bvd_inj
res_inj={}
for sv in [0.0,2.0,5.0]:
    INJ_S=sv; CACHE.clear(); o=amp_stats(1.0,stays100); res_inj[sv]=o
    nl=np.mean([-np.log(max(x[5][fb_(x[1],x[6])],1e-12)) for x in o])
    sh=np.mean([x[4]-x[3] for x in o if x[1]=='glucose_finger' and 780<=x[2]<1500])
    mf=np.array([x[3] for x in o]); d0=mf-np.array([x[3] for x in res_inj[0.0]])
    gmask=np.array([x[1]=='glucose_finger' for x in o])
    print(sv,'nll %.4f GFshift %.2f factual GF move mean %.2f sd %.2f'%(nl,sh,d0[gmask].mean(),d0[gmask].std()))
INJ_S=0.0; CACHE.clear()
```