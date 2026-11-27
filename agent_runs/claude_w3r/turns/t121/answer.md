Assigning mg/dL to unit row 3 and mmHg to row 4 lowers the measurement-bundle error (0.084 to 0.064), but some error remains. My local core step is roughly right: about 3% error on `out`, better on action bundles.

Rather than chase exact replication, I'll test what matters for the edit. The glucose prediction for hour t+1 reads the state just after the t:05 insulin bundle. So a dose channel needs to cross only one core step, from u_k to `out`. Next I will compute, at insulin bundles of s0, the local Jacobian of `out` with respect to u_k. I compare it with the natural spread of `out` across insulin-bundle states (from SX) to find the u-direction with the best signal-to-noise and check whether the Jacobian is stable across bundles.

- **Confirms feasibility:** a u-direction gives a high signal-to-noise response, consistent across bundles.
- **Refutes it:** every response is buried in the state's normal variation.

```python
OPTS['unit'].update({'glucose_finger':3,'glucose_lab':3,'map_arterial':4,'map_cuff':4})
B0=build_bundles(ev0,OPTS); EU0=[enc_eu(al) for _,al in B0]
def step_at(B,rows,EU,k,du=None,de=None):
    r_=rows[k]; hp=torch.tensor(SX[rows[k-1],192:960].reshape(4,192),dtype=torch.float64)
    zk=torch.tensor(SX[r_,960:],dtype=torch.float64); idx_=list(range(k+1))[-128:]
    e_,u_=EU[k]; e_=e_ if de is None else e_+de; u_=u_ if du is None else u_+du
    Mem=torch.stack([EU[j][0] if j!=k else e_ for j in idx_]); Mage=torch.tensor([(B[k][0]-B[j][0])*3600 for j in idx_],dtype=torch.float64)
    y,o,E=core_step_v(hp,(B[k][0]-B[k-1][0])*3600,e_,u_,zk,Mem,Mage,'std'); return y,o
ins_k=[k for k in range(1,len(B0)) if any(a['code']==CODE['insulin'] and a['kind']==0 for a in B0[k][1])]
print('insulin bundles in s0:', ins_k[:20])
eps=1e-3; Js=[]
for k in ins_k[:3]:
    y0,o0=step_at(B0,rows0,EU0,k); J=torch.zeros(192,192,dtype=torch.float64)
    for i in range(192):
        du=torch.zeros(192,dtype=torch.float64); du[i]=eps; _,o1=step_at(B0,rows0,EU0,k,du=du); J[:,i]=(o1-o0)/eps
    Js.append(J.numpy())
OI=SX[z['has_insulin'],:192].astype(np.float64); Sig=np.cov(OI.T); print('n insulin states',OI.shape[0])
Sig_r=Sig+1e-4*np.trace(Sig)/192*np.eye(192); Si=np.linalg.inv(Sig_r)
for J in Js:
    sv=np.linalg.svd(J,compute_uv=False); Mq=J.T@Si@J; ev_=np.linalg.eigvalsh(Mq)[::-1]
    print('J sv top5',sv[:5].round(3),' SNR^2 top5 per unit |du|:',ev_[:5].round(1))
c12=np.sum(Js[0]*Js[1])/np.linalg.norm(Js[0])/np.linalg.norm(Js[1]); c13=np.sum(Js[0]*Js[2])/np.linalg.norm(Js[0])/np.linalg.norm(Js[2])
print('Jacobian similarity across bundles', round(c12,3), round(c13,3))
```