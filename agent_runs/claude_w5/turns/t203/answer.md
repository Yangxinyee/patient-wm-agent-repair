The prelude-to-block route has enough headroom, though not a lot:
- The best direction gives a signal-to-variation ratio of about 1.9 per unit of slot-residual write at the block-1 FFN input (eigenvalue 3.5).
- The prelude LayerNorm then scales the write by about 0.016 into E.
- Its gains are close to 1 (0.94–1.06).

So a carrier of about 15 per unit rate, around 2% of |T| ≈ 760 per u/h, should give a block-1 readout ratio near 28, i.e. about 0.04 rate units of error.

Next I build stage 1 in the working weights. A rank-one change to head 1's value rows reads the rates from the u token, using the ridge readouts that are exactly 0 on the empty-drug token. Head 1's output matrix then carries this into the slots.

I choose the two head-1 directions (insulin, norepinephrine) to:
- give the best block-1 readability;
- be orthogonal to the head-gradient directions, so the carrier does not shift the next glucose or MAP prediction by itself.

The check: at a drug bundle the slot-residual change should be about 15 × rate, and at non-drug bundles nearly 0.

```python
W64o={}
def setW(name,val):
    if name not in W64o: W64o[name]=W64[name].clone()
    W64[name]=val
def resetW():
    for k,v in W64o.items(): W64[k]=v.clone()
gT=torch.tensor(gm,dtype=torch.float64)@Kem          # 2x192 head grads wrt shared dT
B1=WO1; Cn=gT@B1
Ns=torch.linalg.svd(Cn,full_matrices=True)[2][2:].T    # 48x46 null space
Ah=Ns.T@B1.T@Mk@B1@Ns; Bh=Ns.T@B1.T@B1@Ns
L=torch.linalg.cholesky(Bh); Li=torch.linalg.inv(L); ev2,ez=torch.linalg.eigh(Li@Ah@Li.T)
zI=Li.T@ez[:,-1]; zN=Li.T@ez[:,-2]
cI=Ns@zI; cN=Ns@zN; alpha=15.0
cI=cI*alpha/(B1@cI).norm(); cN=cN*alpha/(B1@cN).norm()
print('SNR^2/|dT|^2 in head1 span w/ constraints: %.2f %.2f'%(float(ev2[-1]),float(ev2[-2])),'head-grad leak',np.round((gT@(B1@cI)).numpy(),5))
resetW(); kvn='core.prelude.0.attn.kv.weight'; Wn=W64[kvn].clone()
Wn[240:288]+=torch.outer(cI,QIN[:,0])+torch.outer(cN,QIN[:,1]); setW(kvn,Wn)
dI,dN=B1@cI,B1@cN
chk=[]
for d in [x for x in D if x['sid'] in VAL][:300]:
    r=RK[d['sid']][d['k']]; at=bundles_B[d['sid']][d['k']][1]
    T1=T_post_attn(r,at); resetW_=W64[kvn]; W64[kvn]=W64o[kvn]; T0=T_post_attn(r,at); W64[kvn]=resetW_
    dT=(T1-T0).mean(0); chk.append((isdrug(d),d['ri'],d['rn'],float(dT@dI/alpha**2),float(dT@dN/alpha**2),float(dT.norm())))
chk=np.array(chk); dr=chk[:,0]==1
print('drug: corr(sigI,ri) %.3f slope %.3f | corr(sigN,rn) %.3f slope %.3f'%(np.corrcoef(chk[dr,3],chk[dr,1])[0,1],np.polyfit(chk[dr,1],chk[dr,3],1)[0],np.corrcoef(chk[dr,4],chk[dr,2])[0,1],np.polyfit(chk[dr,2],chk[dr,4],1)[0]))
print('nondrug |dT| mean %.3f max %.3f | drug |dT| mean %.2f'%(chk[~dr,5].mean(),chk[~dr,5].max(),chk[dr,5].mean()))
```