Two results.
- **Policy function:** PF reproduces gg's policy mean (mean absolute difference 0.07 U/h), so the oracle can use PF(g) for the policy compensation.
- **Norepinephrine (arterial MAP, n1..n5 only):** the model misses +2.71 (±0.31) at lag 1, +0.58 at lag 2 and +0.33 at lag 3, and over-predicts at lags 4–5 (−1.91, −1.09). The sum is only about +0.6. So the model's long-run response to a sustained NE level is roughly right, but its response is too slow. Cuff MAP shows the same pattern, a bit weaker (+1.86 at lag 1).

Next, the decisive test before building anything in weights. I run the factual-optimal correction as an exact oracle in my local rollout engine on the 100 train stays (Ro) and compare with the record-based truth for all scored interventions:
- the model's insulin correction: dose lags −1.27, −1.35, −0.63, +0.60, +0.83, policy lags +0.59, +1.19, +0.31, −0.40, −0.18;
- the norepinephrine correction: +2.71, +0.58, +0.33, −1.91, −1.09;
- interventions: x2, x1/2, x0, x3, x1.5, start at hour 16 and at hour 6 (with the record hidden from hour 7).

**Outcomes:** if the rollouts land near the truth (insulin x2 ≈ −16.5), the factual-optimal correction is also counterfactually right and I implement exactly it. If they fall short, the model's own propagation is biased and I will need extra gain, limited by the 0.005 NLL budget.

```python
print('W==orig:', max(float((W[k]-load_original(k)).abs().max()) for k in NAMES))
a_i=[-1.265,-1.349,-0.632,0.595,0.825]; c_i=[0.592,1.190,0.305,-0.404,-0.182]; k_i=0.117
a_n=[2.707,0.580,0.334,-1.914,-1.090]; k_n=-0.028
import time
def run_or(R_,arms,t_hide,**kw):
    DGi=dose_stack(R_,arms,'insulin'); DGn=dose_stack(R_,arms,'norepinephrine')
    hk=make_lin_oracle(R_,DGi,a_i,c_i,k_i,PF,DGn,a_n,k_n,t_hide=t_hide,**kw)
    return run_rollout_h2(R_,W,arms,n_paths=20,seed=0,t_hide=t_hide,hook_P=hk)
t0=time.time()
armsA=[{},{'ins':2.0},{'ins':0.5},{'ins':0.0},{'ins':3.0},{'ins':1.5},{'ins':2.0,'ins_from':16.0},{'ne':2.0},{'ne':0.5},{'ne':0.0}]
resA=run_or(Ro,armsA,13.0)
for o in eval_arms(resA,armsA,Ro,K=Kins,drug='insulin',outcome='glucose_finger')[:6]: print('INS',o)
for o in eval_arms(resA,armsA,Ro,K=Knor,drug='norepinephrine',outcome='map_arterial')[6:]: print('NE',o)
armsB=[{},{'ins':2.0,'ins_from':6.0},{'ne':2.0,'ne_from':6.0}]
resB=run_or(Ro,armsB,7.0)
print('INS6',eval_arms(resB,armsB,Ro,K=Kins,drug='insulin',outcome='glucose_finger')[0])
print('NE6',eval_arms(resB,armsB,Ro,K=Knor,drug='norepinephrine',outcome='map_arterial')[1])
print('time',time.time()-t0)
```