The shrunk LDA gate works very well. At λ = 0.01 or 0.1, every glucose sample projects to +7…+12 and every MAP sample to −8…−13, with within-code spread around 0.2 and a moderate weight norm (about 7). A single linear direction can therefore switch the edit unit fully on for glucose_finger/lab and fully off for MAP, and the reverse for the norepinephrine unit.

Next is the signal: teacher-forced counterfactual states. I will write the runner. It starts from the tool's factual state just before hour 12, then scales every drug atom from hour 12 on (rate and dose, END atoms included, as the dose edit does). It recomputes u_k and z_k, and steps the replica with readings left as recorded.

As a first test on one stay I will check:
- that u_k really changes at drug bundles;
- whether END-atom attributes matter;
- the replica's factual drift against the tool (f = 1);
- the run time.

```python
import time
def mod_events(evs,edits):
    o_=[]
    for (ev,kind,row) in evs:
        f=edits.get(ev,1.0)
        if f!=1.0: row=row._replace(rate=row.rate*f, dose=row.dose*f)
        o_.append((ev,kind,row))
    return o_
def run_cf(s,edits,k_from_min=720,X=None):
    B=BUN[s]; X=X_of(s) if X is None else X; n=len(B)
    k0=next((k for k in range(n) if B[k][0]>=k_from_min),n)
    O=[X[k,:192] for k in range(k0)]; H=[X[k,192:960].reshape(4,192) for k in range(k0)]
    o,h=O[-1],H[-1]
    for k in range(k0,n):
        e,u=bvd(mod_events(B[k][1],edits)); e,u=e.double(),u.double()
        dt=(B[k][0]-B[k-1][0])*60.0; js=list(range(k+1))[-128:]
        Mem=torch.stack([vecs(s,j)[0] for j in js]); ages=torch.tensor([(B[k][0]-B[j][0])*60.0 for j in js],dtype=torch.float64)
        z=zpost(o,h,dt,e,u); o,h=step(o,h,dt,e,u,z,Mem,ages); O.append(o); H.append(h)
    return torch.stack(O),torch.stack(H),k0
t0=time.time(); O1,H1,k0=run_cf(s0,{}); t1=time.time(); O2,H2,_=run_cf(s0,{'insulin':2.0}); t2=time.time()
X0=X_of(s0)
print('k0',k0,'n',len(BUN[s0]),'time',round(t1-t0,2),round(t2-t1,2))
print('factual replica drift max', round(float((O1-X0[:,:192]).abs().max()),4), 'cf-fact out diff max', round(float((O2-O1).abs().max()),4))
dk=[k for k in range(k0,len(BUN[s0])) if any(a[0]=='insulin' for a in BUN[s0][k][1])]
print('ins bundles',len(dk),[ [a[1] for a in BUN[s0][k][1] if a[0]=='insulin'] for k in dk[:4]])
```