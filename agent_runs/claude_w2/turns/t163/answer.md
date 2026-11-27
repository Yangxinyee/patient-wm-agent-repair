The paired regressions (12k glucose_finger rows) show where the model goes wrong:

| | AR coefficients | insulin lags 1–3 | lag-1 policy term |
|---|---|---|---|
| records | .47 / .32 / .20 | −2.08, −1.56, −0.46 | −0.36 |
| model | similar | −0.52, −0.13, +0.21 | −1.53 |

So the model uses the policy dose implied by the last glucose as a stand-in for the actual dose. Its response to the typical dose is roughly right, but its response to deviations from the policy is about 4× too weak. For MAP, the model's NE lag-1 coefficient is ≈0 against +3.1 in the records, while its MAP persistence is about right. The NE lag coefficients are collinear (random-walk rates), so the smoother structural kernel is the better guide there.

The correction therefore has to read recent doses and the policy dose from the state that the value head sees. Before building it, I'll test that this is feasible. I'll run teacher-forced passes on 600 insulin stays under several dose arms (×2 from 12, ×½ from 6, ×0 from 12, ×3 from 16), keep the factual glucose, and store the head's input features at every reading. The question is how well a linear decoder recovers the insulin rates at lags 1–3 and the policy dose at lags 1–2, and whether it responds 1:1 to counterfactual dose changes.

```python
code_roll2=code_roll.replace("            P=P/4\n","            P=P/4\n            if store is not None: store.append((hn_.copy(),hj.copy(),base[inv].clone(),P.clone(),f8.clone(),lb_.clone(),zmu[inv].clone()))\n")
code_roll2=code_roll2.replace("def run_rollout(R,W,arms,n_paths=20,seed=0,t_hide=13.0,want_nll=False,chunk=None):","def run_rollout(R,W,arms,n_paths=20,seed=0,t_hide=13.0,want_nll=False,chunk=None,store=None):")
open('/work/rollout.py','w').write(code_roll2); exec(code_roll2)
ins_stays=sorted(tr[tr.event=='insulin'].stay_id.unique()); rng_=np.random.default_rng(7)
idsI=sorted(rng_.choice(ins_stays,600,replace=False).tolist())
armsI=[{},{'ins':2.0,'ins_from':12.0},{'ins':0.5,'ins_from':6.0},{'ins':0.0,'ins_from':12.0},{'ins':3.0,'ins_from':16.0}]
RI=prep_rollout(tr,idsI); stI=[]
t0=time.time(); resI=run_rollout(RI,W,armsI,n_paths=1,seed=0,t_hide=1e9,want_nll=True,store=stI); print('time',time.time()-t0)
hnI=np.concatenate([s[0] for s in stI]); hjI=np.concatenate([s[1] for s in stI]); baseI=torch.cat([s[2] for s in stI]); PI=torch.cat([s[3] for s in stI])
f8I=torch.cat([s[4] for s in stI]); lbI=torch.cat([s[5] for s in stI]); zmuI=torch.cat([s[6] for s in stI])
print(len(hnI), baseI.shape)
```