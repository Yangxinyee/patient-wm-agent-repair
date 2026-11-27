Two findings.
1. My synthetic u builder differed from the real encoder (max difference 0.18) because I gave norepinephrine a dose equal to its rate. In the data the norepinephrine dose is 4.8 × rate, while insulin's dose equals its rate. I'll fix that.
2. A linear readout of u cannot recover the insulin rate over the doubled range up to 12 u/h. The bins and the z clipping saturate, so the fit needs huge coefficients, or else leaves residual sd 0.6 to 1.2. The rate therefore has to be written into u linearly by editing the encoder.

encode_action has no dead units, so I'll add a rank-1 readout on its mlp2 hidden layer instead. Its 768 GELU units should give a rich enough basis to decode each drug's rate linearly.

Next I'll test that decoding on the full grid. I'll also compute the normal-variation spectrum of E across real bundles, since a low-variance E direction is needed to amplify the signal into h without amplifying noise.

What would confirm the plan:
- the insulin rate is decodable from the mlp2 hidden activations with residual sd under about 0.05;
- E has directions with very small normal variance.

```python
def atom_vec(code,rate,kind=0):
    a=dict(code=CODE[code],kind=kind,ek=2,unit=OPTS['unit'][code],ru=OPTS['ru'][code],val=None,dose=(code,rate*(4.8 if code=='norepinephrine' else 1.0)),rate=(code,rate))
    return mm.embed_atom(a,OPTS).double()
print('u_of vs enc_eu max diff (fixed)', float((u_of(ri[0] if ri else None, rn[0] if rn else None)-EU0[kk][1]).abs().max()))
def s_of(ins=None, ne=None):
    act=[]
    if ins is not None: act.append(atom_vec('insulin',ins))
    if ne is not None: act.append(atom_vec('norepinephrine',ne))
    X_=torch.stack(act); n_=len(act)
    return mlp_t(X_,'encode_action.mlp1').sum(0)/math.sqrt(n_)+(torch.tensor([[math.log(1+n_)]],dtype=torch.float64)@Wt['encode_action.size.weight'].T+Wt['encode_action.size.bias'])[0]
Sg=[];Ti=[];Tn=[]
for r in gi: Sg.append(s_of(ins=r)); Ti.append(r); Tn.append(0)
for r in gn: Sg.append(s_of(ne=r)); Ti.append(0); Tn.append(r)
for r in gi:
    for q_ in gn[::2]: Sg.append(s_of(ins=r,ne=q_)); Ti.append(r); Tn.append(q_)
Sg=torch.stack(Sg); Ti=np.array(Ti); Tn=np.array(Tn)
H2=F.gelu(Sg@Wt['encode_action.mlp2.0.weight'].T+Wt['encode_action.mlp2.0.bias']).numpy()
H2b=np.hstack([H2,np.ones((H2.shape[0],1))])
for nm_,T_ in [('insulin',Ti),('norepi',Tn)]:
    for lam in [1e-6,1e-4,1e-2]:
        cvec=np.linalg.solve(H2b.T@H2b+lam*np.eye(H2b.shape[1]),H2b.T@T_); res=H2b@cvec-T_
        print(nm_,'lam',lam,' resid sd %.4f  max %.3f  |a| %.1f'%(res.std(),np.abs(res).max(),np.linalg.norm(cvec[:-1])))
# normal E spectrum (mean over slots) from rowsE-like computation over all bundles of 20 stays
EM=[]
for s in selS[:20]:
    ev_=recs[recs.stay_id==s].sort_values('hour'); B_=build_bundles(ev_,OPTS); rw=np.where(SST==s)[0]
    if len(B_)!=len(rw): continue
    EU_=[enc_eu(al) for _,al in B_]
    for k in range(1,len(B_)):
        hp=torch.tensor(SX[rw[k-1],192:960].reshape(4,192),dtype=torch.float64); zk=torch.tensor(SX[rw[k],960:],dtype=torch.float64)
        idx_=list(range(k+1))[-128:]; Mem=torch.stack([EU_[i][0] for i in idx_]); Mage=torch.tensor([(B_[k][0]-B_[i][0])*3600 for i in idx_],dtype=torch.float64)
        _,_,E=core_step_v(hp,(B_[k][0]-B_[k-1][0])*3600,EU_[k][0],EU_[k][1],zk,Mem,Mage,'std'); EM.append(E.mean(0).numpy())
EM=np.array(EM); evE=np.linalg.eigvalsh(np.cov(EM.T))[::-1]; print('n E', EM.shape[0], ' E var eig (sqrt) top5', np.sqrt(evE[:5]).round(3), ' bottom10', np.sqrt(np.maximum(evE[-10:],0)).round(4))
```