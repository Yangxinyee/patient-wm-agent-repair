This is a decisive result. Through the prelude, `out` responds to u_k in essentially one direction only (top singular value 0.05, the next 0.001). The four slots therefore barely attend to the u_k token; that single direction is just a side effect of attention weights shifting. So u_k itself is almost a dead end. The dose must reach the state mainly through the latent z: the posterior z reads u_k, and z then enters the prelude as the z_in token. That fits my earlier finding that z dims 9 and 20 encode the insulin rate.

So the channel should go through z. Next I will compute, at the same insulin bundles, the Jacobian of `out` with respect to the 32 posterior z dimensions and its signal-to-noise against normal state variation. I will also check how much each z dim's posterior mean varies across bundles, to spot inactive dims that could be repurposed.

- **Confirms:** some z dims (ideally rarely used ones) move `out` strongly and stably.
- **Refutes:** the z responses are also weak. Then I would have to work through attention weights or memory.

```python
Jz=[]
for k in ins_k[:3]:
    r_=rows0[k]; y0,o0=step_at(B0,rows0,EU0,k); J=np.zeros((192,32))
    for j in range(32):
        hp=torch.tensor(SX[rows0[k-1],192:960].reshape(4,192),dtype=torch.float64)
        zk=torch.tensor(SX[r_,960:],dtype=torch.float64).clone(); zk[j]+=1e-3
        idx_=list(range(k+1))[-128:]; Mem=torch.stack([EU0[i][0] for i in idx_]); Mage=torch.tensor([(B0[k][0]-B0[i][0])*3600 for i in idx_],dtype=torch.float64)
        _,o1=core_step_v(hp,(B0[k][0]-B0[k-1][0])*3600,EU0[k][0],EU0[k][1],zk,Mem,Mage,'std'); J[:,j]=((o1-o0)/1e-3).numpy()
    Jz.append(J)
Zall=SX[:,960:].astype(np.float64); zsd=Zall.std(0); zsd_ins=SX[z['has_insulin'],960:].std(0)
gain=np.linalg.norm(Jz[0],axis=0); snr=np.sqrt(np.einsum('ij,ik,kj->j',Jz[0],Si,Jz[0]))
zin_n=np.linalg.norm(load_original('z_in.weight'),axis=0); zh_n=np.linalg.norm(load_original('z_head.weight'),axis=0)
print('dim  sd_all  sd_ins  |dout/dz|  SNR/unit  |z_in col| |z_head col|')
for j in np.argsort(-snr): print(j, round(zsd[j],3), round(zsd_ins[j],3), round(gain[j],3), round(snr[j],2), round(zin_n[j],2), round(zh_n[j],2))
cs=[np.sum(Jz[0]*Jz[i])/np.linalg.norm(Jz[0])/np.linalg.norm(Jz[i]) for i in (1,2)]; print('Jz similarity', np.round(cs,3))
```