The residual regressions decide the compensation design.

- **Finger glucose (n = 24,386):** the original model misses the effect of insulin deviations from the policy. Lag coefficients are −1.20, −1.42, −0.62, +0.61 (se ≈ 0.16). The policy-predicted part leaves only a small residual dependence (−0.54, −0.18, −0.20, +0.49). So the model has folded the average, policy-driven insulin effect into its glucose dynamics, and the correction should act on deviations.
  - The absolute-dose regression gave positively signed coefficients, which looks like a leak from a different lag convention in that column. I'll ignore it.
- **Arterial MAP:** the residual on absolute norepinephrine lags is +2.69, −0.49, −0.34, −2.33. The model misses the effect of recent changes in norepinephrine but is calibrated for its level, which MAP history already carries.

This gives a concrete hypothesis. If I add exactly these missing one-step terms to the model's predictions, the model's own propagation of drawn values should reproduce the true counterfactual effects. For insulin the terms are Σ a·d + Σ (c − a)·policy(g), in absolute doses and policy-of-glucose lags; for norepinephrine they are Σ b·n.

I'll test this with the exact linear oracle in local rollouts on the 100 train stays, alongside the unedited baseline. The comparison is against the kernel truths. Ratios near 1 across x2, x½, x0, x3 and start-at-16 would confirm the design. A uniform undershoot would mean I need an extra gain; arm-dependent ratios would mean the propagation is wrong.

```python
t0=time.time()
Rx=Ro; print('stays',len(Rx['stays']))
arms_i=[{},{'ins':2.0},{'ins':0.5},{'ins':0.0},{'ins':3.0},{'ins':2.0,'ins_from':16.0}]
arms_n=[{},{'ne':2.0},{'ne':0.5}]
DGi_=dose_stack(Rx,arms_i,'insulin'); DGn_=dose_stack(Rx,arms_n,'norepinephrine')
print('NE grid mean nonzero', DGn_[0][DGn_[0]>0].mean().round(3))
a_i=[-1.199,-1.415,-0.624,0.606]; c_i=[0.655,1.236,0.428,-0.112]; a_n=[2.686,-0.491,-0.341,-2.329]
res_b=run_rollout_h2(Rx,W,arms_i,n_paths=10,seed=1)
print('BASE', [x[:3] for x in eval_arms(res_b,arms_i,Rx,Kins)])
hk_i=make_lin_oracle(Rx,DGi=DGi_,a_i=a_i,c_i=c_i,k_i=0.164,pf=pfun)
res_l=run_rollout_h2(Rx,W,arms_i,n_paths=10,seed=1,hook_P=hk_i)
for x in eval_arms(res_l,arms_i,Rx,Kins): print('LIN', x)
hk_n=make_lin_oracle(Rx,DGn=DGn_,a_n=a_n,k_n=0.303)
res_n=run_rollout_h2(Rx,W,arms_n,n_paths=10,seed=1,hook_P=hk_n)
for x in eval_arms(res_n,arms_n,Rx,Knor,drug='norepinephrine',outcome='map_arterial'): print('LIN-NE', x)
res_nb=run_rollout_h2(Rx,W,arms_n,n_paths=10,seed=1)
print('BASE-NE', [x[:3] for x in eval_arms(res_nb,arms_n,Rx,Knor,drug='norepinephrine',outcome='map_arterial')], round(time.time()-t0))
```