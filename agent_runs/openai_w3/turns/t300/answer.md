The caches now match, and the original reference is valid. I’ll fit a linear feed-forward correction that preserves the current factual update while making the **total** changed-dose update follow the record-derived response directions. Held-out stays will test whether this restores dose contrasts without disrupting factual states.

```python
import numpy as np, torch, scipy.linalg as sla
torch.set_num_threads(8)
checkpoint267 = {n: load(n).clone() for n in edited()}
mu267 = np265(pcur265[1]).reshape(-1,768).mean(axis=0).astype(np.float64)
sd267 = np.maximum(np265(pcur265[1]).reshape(-1,768).std(axis=0),0.02).astype(np.float64)
H267 = np265(pcur265[1]).reshape(-1,768).astype(np.float64)
X267 = np.column_stack(((H267-mu267)/sd267, np.ones(len(H267))))
unique267 = np.unique(np265(stay88))
train_b267 = np.isin(np265(stay88), unique267[:70])
tr267 = np.repeat(train_b267,4)
va267 = ~tr267
Cf267 = X267[tr267].T @ X267[tr267] / tr267.sum()
Cd267 = np.zeros((769,769),np.float64)
Rd267 = np.zeros((769,192),np.float64)
contrast267 = []
dirs267 = np.asarray(directions255,dtype=np.float64)
rates267 = np.asarray(np265(rate88),dtype=np.float64)
base_raw267 = np265(pcur265[2]).reshape(-1,192).astype(np.float64)
print("held-out stays", len(unique267)-70, "active bundles", (rates267>0).sum(axis=0).tolist())
for (drug,scale),tok in sorted(fresh_tokens262.items()):
    mask = np.repeat(rates267[:,drug]>0,4)
    pp = parts254(tok)
    dh = (np265(pp[1]).reshape(-1,768).astype(np.float64)-H267)/sd267
    dx = np.column_stack((dh[mask],np.zeros(mask.sum())))
    actual = np265(pp[2]).reshape(-1,192).astype(np.float64)[mask]-base_raw267[mask]
    dr = np.repeat(rates267[:,drug]*(scale-1),4)[mask]
    desired = dr[:,None]*dirs267[:,drug][None,:]
    target = desired-actual
    tm = tr267[mask]
    vm = ~tm
    Cd267 += (dx[tm].T@dx[tm])/tm.sum()/8
    Rd267 += (dx[tm].T@target[tm])/tm.sum()/8
    contrast267.append((drug,scale,dx,actual,desired,tm,vm))
    coeff = np.mean(actual.reshape(-1,4,192),axis=1) @ np.linalg.pinv(dirs267).T
    drb = dr.reshape(-1,4)[:,0]
    print("contrast",drug,scale,"carrier slopes",
          np.round((drb[:,None]*coeff).sum(axis=0)/(drb@drb),3).tolist())
w267 = np265(load("core.prelude.0.ffn.2.weight")).astype(np.float64)
b267 = np265(load("core.prelude.0.ffn.2.bias")).astype(np.float64)
ng267 = np265(load("core.prelude_norm.weight"))
nb267 = np265(load("core.prelude_norm.bias"))
base_E267 = np265(pcur265[3]).reshape(-1,192)
def normalize267(a):
    return (a-a.mean(axis=-1,keepdims=True))/np.sqrt(a.var(axis=-1,keepdims=True)+1e-5)*ng267+nb267
candidates267 = {}
penalty267 = np.eye(769)*0.001
penalty267[-1,-1] = 1e-8
print("gamma | factual raw RMS | held-out normalized RMS | held-out contrast relative error")
for gamma in [0.1,1.0,10.0,100.0]:
    B = sla.cho_solve(sla.cho_factor(Cf267+gamma*Cd267+penalty267,lower=True),gamma*Rd267)
    delta = X267@B
    er = np.sqrt(np.mean((normalize267(base_raw267+delta)[va267]-base_E267[va267])**2))
    num=den=0.0
    for drug,scale,dx,actual,desired,tm,vm in contrast267:
        num += np.mean((actual[vm]+dx[vm]@B-desired[vm])**2)
        den += np.mean(desired[vm]**2)
    dw = B[:768].T/sd267
    wn = (w267+dw).astype(np.float32)
    bn = (b267+B[-1]-dw@mu267).astype(np.float32)
    candidates267[gamma] = {"weight":wn,"bias":bn,"B":B,"E_rmse":float(er),
                            "contrast_relative":float(np.sqrt(num/den))}
    print(gamma,round(float(np.sqrt(np.mean(delta[va267]**2))),4),
          round(float(er),5),round(float(np.sqrt(num/den)),4))
with open("/work/notes.md","a") as f:
    f.write("\nPython run 267; forwards 20. Fitted total-raw-response linear corrections from exact factual and x0/.5/2/3 tokens, using record-derived full-sensor directions and whole-stay holdout. Factual target is zero change from validated checkpoint222. Candidates retained, not installed. Scores: "+
            str([(g,c["E_rmse"],c["contrast_relative"]) for g,c in candidates267.items()])+"\n")
```