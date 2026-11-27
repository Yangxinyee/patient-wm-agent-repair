The insulin direction behaves reasonably near typical doses, but fitting all four sensors made the norepinephrine direction large and nonlinear. I’ll prioritize the primary readouts and refit the carrier using paired contrasts at every multiplier. Small factual residuals and accurate held-out contrasts would justify an authoritative pilot.

```python
import numpy as np, torch, time
t187 = time.time()

# Prioritize the two main readouts without forcing an ill-conditioned sensor ratio.
Jmain187 = J186[[1,2], :]
Tmain187 = np.array([[-1.94,0.0],[0.0,3.2]])
directions187 = Jmain187.T @ np.linalg.solve(
    Jmain187 @ Jmain187.T + 1e-9*np.eye(2), Tmain187)
print("Primary direction norms", np.linalg.norm(directions187,axis=0))
print("All-sensor linear responses\n", J186@directions187)
for drug,gains in [(0,[-2,-1,-.5,.5,1,2]),
                   (1,[-.75,-.5,-.25,.125,.25,.5,.75])]:
    g = np.array(gains)
    print("Drug",drug,"physical-rate changes",g,
          "\nsensor shifts\n",
          np.round(response186(g[:,None]*directions187[:,drug][None,:])-base186,4))

rates187 = np.zeros((len(idx81),2))
for local,global_index in enumerate(idx81):
    for row,is_end in r40_info[int(global_index)]:
        if not is_end and row.event in ("insulin","norepinephrine"):
            j = ("insulin","norepinephrine").index(row.event)
            rates187[local,j] += float(row.rate)

post187 = {k:tensor153(v) for k,v in post154.items()}
def latent187(u):
    z = posterior153(C154,E154,u,post187)
    raw = torch.nn.functional.linear(z,tensor153(zin154),tensor153(zb154))
    return arr183(layer_norm(raw,"core.prelude.0.norm1"))

x0_187 = latent187(U154).astype(np.float64)
train187 = np.asarray(stays177)%5 != 0
valid187 = ~train187
gate187 = np.asarray(gatevec177,dtype=np.float64)
mu187 = x0_187[train187].mean(0)
xc187 = x0_187 - (x0_187@gate187)[:,None]*mu187[None,:]
pairs187 = {}
for drug in [0,1]:
    for factor in [0.,.5,1.5,2.,3.]:
        factors = [1.,1.]
        factors[drug] = factor
        xs = latent187(action_vectors87(tuple(factors))).astype(np.float64)
        dx = xs-x0_187
        dc = dx - (dx@gate187)[:,None]*mu187[None,:]
        target = np.zeros_like(rates187)
        target[:,drug] = (factor-1)*rates187[:,drug]
        pairs187[(drug,factor)] = (dc,target)

A187 = np.zeros((192,192))
rhs187 = np.zeros((192,2))
for dc,y in pairs187.values():
    A187 += dc[train187].T@dc[train187]/train187.sum()/len(pairs187)
    rhs187 += dc[train187].T@y[train187]/train187.sum()/len(pairs187)
Cf187 = xc187[train187].T@xc187[train187]/train187.sum()
decoders187 = {}
for penalty in [.1,1.,10.]:
    A = A187 + penalty*Cf187
    ridge = 1e-8*np.trace(A)/192
    B = np.linalg.solve(A + ridge*np.eye(192),rhs187)
    B = B - gate187[:,None]*(mu187@B)[None,:]
    decoders187[penalty] = B
    print("\nFactual penalty",penalty,"decoder norms",np.linalg.norm(B,axis=0),
          "held-out factual RMS",
          np.sqrt(np.mean((x0_187[valid187]@B)**2,axis=0)))
    for drug,factor in [(0,0.),(0,.5),(0,2.),(0,3.),
                        (1,0.),(1,.5),(1,2.),(1,3.)]:
        dc,y = pairs187[(drug,factor)]
        active = valid187 & (rates187[:,drug]>0)
        prediction = dc@B
        print(drug,factor,"target/predicted mean",
              round(float(y[active,drug].mean()),4),
              round(float(prediction[active,drug].mean()),4),
              "RMS",np.round(np.sqrt(np.mean((prediction[active]-y[active])**2,axis=0)),4))

value_norms187 = np.linalg.norm(arr183(zero169["core.prelude.0.attn.kv.weight"])[336:384],axis=1)
print("\nReserved value row norms",np.round(value_norms187,5))
print("Latent tensor names",[n for n in NAMES if "z_post" in n or "z_in" in n])
print("Available stays",len(np.unique(stays177)),np.unique(stays177)[:12])
print("Elapsed seconds",round(time.time()-t187,2))

with open("/work/notes.md","a") as f:
    f.write("\nPython run 187; forwards 10. Four-sensor calibration is nonlinear/ill-conditioned for NE; prioritize glucose_finger and map_arterial, retain alternate-sensor diagnostics. Refit full normalized-latent linear carriers on paired physical-dose contrasts at 0/.5/1.5/2/3 with factual covariance penalties and whole-stay holdout. No new edits yet.\n")
```
