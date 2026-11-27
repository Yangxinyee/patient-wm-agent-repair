Run 225 shows that no replica function reads any of the other names I reused, so restoring `F` repairs the kernel. Next I will rerun the amplification experiment unchanged: s = 0, 2 and 5 at insulin bundles, reporting factual NLL, the teacher-forced insulin x2 shift and the factual glucose movement. As before, a shift that grows with (1+s) supports the idea, and one stuck near −0.9 refutes it.

```python
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