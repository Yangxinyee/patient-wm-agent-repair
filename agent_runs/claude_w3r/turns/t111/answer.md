The run stopped on a dtype clash: my encoder builds a float32 numeric vector while I had converted its weights to float64. Everything defined before line 49 is in the kernel. I will keep the encoder in float32, convert its outputs e and u to float64, and rerun the stay alignment and the variant scan. The confirm and refute criteria are unchanged.

```python
mm=M({n:load_original(n) for n in NAMES})
def enc_eu(atoms):
    obs=[a for a in atoms if not (a['ek']==2 and a['kind']==0)]; act=[a for a in atoms if a['ek']==2 and a['kind']==0]
    e=mm.encode_set(torch.stack([mm.embed_atom(a,OPTS) for a in obs]),'encode').double() if obs else torch.zeros(192,dtype=torch.float64)
    u=mm.encode_set(torch.stack([mm.embed_atom(a,OPTS) for a in act]),'encode_action').double() if act else torch.zeros(192,dtype=torch.float64)
    return e,u
B0,rows0,EU0=stay_data(s0); print('n bundles',len(B0),'rows',len(rows0), 'times match', len(B0)==len(rows0) and np.allclose([t for t,_ in B0], SHR[rows0], atol=1e-3))
res_=[]; t0=time.time()
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
    res_.append((round(np.median(errs),4),round(np.max(errs),4),scn,kf,mn,incl))
res_.sort(); print('time',round(time.time()-t0,1))
for rr_ in res_[:8]: print(rr_)
print('worst', res_[-1])
```