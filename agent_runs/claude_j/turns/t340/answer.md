The bundle format is clear: each event is a tuple (name, kind, record). The helper `bvd(events)` returns (e, u). So I can rebuild u with a drug's rate and dose rescaled, and measure how the head‑2 input `norm1(u)` changes.

Plan:
1. For every drug bundle in the 100 sel1 stays, compute the change in `norm1(u)` when that drug's dose is ×2 and ×½.
2. Take the top directions of those changes, made orthogonal to the mean token and to the empty‑token vector, so constant parts are not amplified.
3. Amplify head 2's value only along those directions, with a separate gain per drug (test value 12).
4. Print how much of head 2's dose‑induced value change the chosen directions reproduce (β), and the overlap between the two drugs.

Expected: β near 1 with few directions. Then the tool should show a response similar to uniform scaling at the same k, at lower factual cost.

```python
import time
t0=time.time()
n1w=A(load_original('core.prelude.0.norm1.weight')).astype(np.float64); n1b=A(load_original('core.prelude.0.norm1.bias')).astype(np.float64)
def ln1(u):
    u=A(u).astype(np.float64); return (u-u.mean())/np.sqrt(u.var()+1e-5)*n1w+n1b
def scl(evs,drug,m):
    out=[]
    for ev in evs:
        if ev[0]==drug:
            r=ev[2]; d={}
            if r.rate==r.rate: d['rate']=r.rate*m
            if r.dose==r.dose: d['dose']=r.dose*m
            out.append((ev[0],ev[1],r._replace(**d))+tuple(ev[3:]))
        else: out.append(ev)
    return out
UF=[]; DL={'insulin':[],'norepinephrine':[]}
for s in BUN:
    for (mn,evs) in BUN[s]:
        has=[d for d in DL if any(ev[0]==d for ev in evs)]
        if not has: continue
        lf=ln1(bvd(evs)[1]); UF.append(lf)
        for d in has:
            for m in (2.0,0.5): DL[d].append(ln1(bvd(scl(evs,d,m))[1])-lf)
print('n',len(UF),{san(d)[:3]:len(v) for d,v in DL.items()},'sec',round(time.time()-t0))
Wv2=K0[288:336].astype(np.float64); mu_u=np.mean(UF,0)
Q0,_=np.linalg.qr(np.stack([mu_u,n1b],1))
po=lambda X,Q: X-(X@Q)@Q.T
def beta(D,R):
    a=(D@R)@(Wv2@R).T; b=D@Wv2.T; return round(float((a*b).sum()/(b*b).sum()),3)
VT={}
for d in DL:
    D=np.array(DL[d]); VT[d]=np.linalg.svd(po(D,Q0),full_matrices=False)[2]
    print(san(d)[:3],'beta r1-4',[beta(D,VT[d][:r].T) for r in (1,2,3,4)])
def bases(rA,rB):
    RA=VT['insulin'][:rA].T
    RB,_=np.linalg.qr(po(VT['norepinephrine'][:rB],RA).T)
    return RA,RB
RA,RB=bases(2,2)
DA=np.array(DL['insulin']); DB=np.array(DL['norepinephrine'])
print('cross A on B',beta(DB,RA),'B on A',beta(DA,RB),'own',beta(DA,RA),beta(DB,RB))
def mk2(kA,kB):
    W2=Wv2+(kA-1)*(Wv2@RA)@RA.T+(kB-1)*(Wv2@RB)@RB.T
    kv=K0.copy(); kv[288:336]=W2.astype(np.float32); save_edit(KV,TT(kv))
mk2(12,12); print('edited',edited())
with open('/work/notes.md','a') as f:
    f.write("- fwd8/9: head2 value x4 (uniform): A x2 -2.65->-4.15, half 1.27->1.97, B x2 .745->.858, half -.435->-.619; factual +0.0011 (GF+.0049 GL+.0044 MA-.0014 MC-.0026).\n- run292: targeted: amplify head2 value only along top dirs RA/RB of d ln1(u) under dose x2/x.5 (orth to mean u and LN bias). mk2(kA,kB). Test mk2(12,12).\n")
```