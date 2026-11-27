The design experiment worked well.
- **Best direction:** the best combination of head-2 value directions moves the value-head input (with_code) by about 13 standard deviations of the factual spread per unit of injected rate. The predicted reading means shift by only about 0.45 RMS per unit of injection.
- **Persistence:** the signal persists, staying large at readings hours after the last drug bundle. The channel therefore acts like a slowly decaying dose accumulator, which is what a lagged drug effect needs.

So a strong, clean channel is possible with very small factual side effects.

Next I'll build it for both drugs, using the top two eigen-directions so the two channels stay separable: p_I for drug I and p_N for drug N. Each is scaled so the per-unit signal is about 4 factual standard deviations. Then, on all 100 stays, I'll compute replica states for four conditions: original; injected (factual); injected with drug I x2 from hour 12; injected with drug N x2 from hour 12. I'll measure:
- whether the response is linear at this scale;
- the factual disturbance, as replica shifts in predicted means and bin NLL by event type;
- the channel's response to the dose edits.

Disturbance well below 0.001 nats with a clear dose response would confirm the channel. Large nonlinearity or NLL shifts would refute it.

```python
lam2,V2=eigh(G,Q+1e-4*np.trace(Q)/48*np.eye(48)); pI=V2[:,-1]; pN=V2[:,-2]
print('|p|',round(float(np.linalg.norm(pI)),2),round(float(np.linalg.norm(pN)),2),'snrI',round(float(np.sqrt(pI@G@pI)),2),'snrN',round(float(np.sqrt(pN@G@pN)),2))
cI=4/np.sqrt(pI@G@pI); cN=4/np.sqrt(pN@G@pN)
wN=torch.as_tensor(PRB['norepinephrine'],dtype=dty)
def inj_W(cI,cN):
    Wn=Wkv0.clone(); Wn[288:336]+=cI*torch.outer(torch.as_tensor(pI,dtype=dty),wI)+cN*torch.outer(torch.as_tensor(pN,dtype=dty),wN); return Wn
WkvI=inj_W(cI,cN); t0=time.time(); DAT={}
try:
    for s in stays:
        rt=RTs[s]; ev=rt['event'].values; ob=rt['observed'].values
        eps_=torch.randn(len(rt),4,32,generator=torch.Generator().manual_seed(int(s)%100000),dtype=torch.float64)
        W[KV]=Wkv0; O,H,_=run_cf(s,{},BUN[s][1][0],X_of(s)); wc0,lg0=hf2(O,H,rt,eps_)
        W[KV]=WkvI; O1,H1,_=run_cf(s,{},BUN[s][1][0],X_of(s)); wc1,lg1=hf2(O1,H1,rt,eps_)
        Xi=torch.cat([T64(O1),T64(H1).reshape(len(O1),-1),T64(X_of(s))[:,960:992]],1)
        cf={}
        for dr in ('insulin','norepinephrine'):
            Oc,Hc,_=run_cf(s,{dr:2.0},720,Xi); cf[dr]=hf2(Oc,Hc,rt,eps_)
        DAT[s]=dict(ev=ev,ob=ob,wc0=wc0,wc1=wc1,lg0=lg0,lg1=lg1,cf=cf,eps=eps_)
finally: W[KV]=Wkv0
print('time',round(time.time()-t0,1))
def bnll(lg,ev,ob):
    p=torch.softmax(torch.as_tensor(lg),-1).mean(1).numpy(); return np.array([-np.log(p[i,int(fb_(ev[i],ob[i]))]+1e-12) for i in range(len(ev))])
EV=np.concatenate([DAT[s]['ev'] for s in stays])
dm=np.concatenate([mean_of(DAT[s]['lg1'],DAT[s]['ev'])-mean_of(DAT[s]['lg0'],DAT[s]['ev']) for s in stays])
dn=np.concatenate([bnll(DAT[s]['lg1'],DAT[s]['ev'],DAT[s]['ob'])-bnll(DAT[s]['lg0'],DAT[s]['ev'],DAT[s]['ob']) for s in stays])
for e_ in np.unique(EV): i=EV==e_; print(e_, i.sum(), 'dmean rms',round(float(np.sqrt((dm[i]**2).mean())),3),'dnll',round(float(dn[i].mean()),5))
print('dnll all',round(float(dn.mean()),5))
```