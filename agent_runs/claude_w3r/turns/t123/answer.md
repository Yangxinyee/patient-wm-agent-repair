The run failed only because `core_step_v` returns three values (y, out, E) and I unpacked two. I fix the unpacking and rerun the same z-Jacobian analysis. The question and the criteria are unchanged: does any z dimension, ideally a rarely used one, move `out` strongly and stably enough to carry a dose channel?

```python
Jz=[]
for k in ins_k[:3]:
    r_=rows0[k]; y0,o0=step_at(B0,rows0,EU0,k); J=np.zeros((192,32))
    hp=torch.tensor(SX[rows0[k-1],192:960].reshape(4,192),dtype=torch.float64)
    idx_=list(range(k+1))[-128:]; Mem=torch.stack([EU0[i][0] for i in idx_]); Mage=torch.tensor([(B0[k][0]-B0[i][0])*3600 for i in idx_],dtype=torch.float64)
    for j in range(32):
        zk=torch.tensor(SX[r_,960:],dtype=torch.float64).clone(); zk[j]+=1e-3
        _,o1,_=core_step_v(hp,(B0[k][0]-B0[k-1][0])*3600,EU0[k][0],EU0[k][1],zk,Mem,Mage,'std'); J[:,j]=((o1-o0)/1e-3).numpy()
    Jz.append(J)
Zall=SX[:,960:].astype(np.float64); zsd=Zall.std(0); zsd_ins=SX[z['has_insulin'],960:].std(0)
gain=np.linalg.norm(Jz[0],axis=0); snr=np.sqrt(np.einsum('ij,ik,kj->j',Jz[0],Si,Jz[0]))
zin_n=np.linalg.norm(load_original('z_in.weight'),axis=0); zh_n=np.linalg.norm(load_original('z_head.weight'),axis=0)
print('dim  sd_all  sd_ins  |dout/dz|  SNR/unit  |z_in col| |z_head col|')
for j in np.argsort(-snr): print(j, round(zsd[j],3), round(zsd_ins[j],3), round(gain[j],3), round(snr[j],2), round(zin_n[j],2), round(zh_n[j],2))
cs=[np.sum(Jz[0]*Jz[i])/np.linalg.norm(Jz[0])/np.linalg.norm(Jz[i]) for i in (1,2)]; print('Jz similarity', np.round(cs,3))
```