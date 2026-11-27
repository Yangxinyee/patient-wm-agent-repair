All context variants stay at 80–110% relative error, even for reading-only bundles. So the measurement encoder or the context is still wrong in a way I can't pin down cheaply. I'll park the core replica. It has cost eight runs without converging, and an upstream edit designed on a wrong replica would be worse than none.

Back to the readout route, with one improvement. The dosing rule is a step function of the glucose reading. The value head already sees that reading through its register: the last-value bin embedding and z-score. My earlier decoder used only the query vector, so it couldn't represent the steps. If I add the last-bin one-hot to the decoder inputs, the lag-1 innovation should decode better.

What matters for the counterfactual is the slope of the decoded value on the true innovation: how strongly the readout moves when the dose moves with the reading held fixed. I'll test this on the existing tuning states, so no forward pass is needed.
- A slope of about 0.4 or more would make the readout edit affordable within the factual budget.
- About 0.2 would not.

```python
sub_=fa[mG].copy()
sub_['nu1']=[Ptu2.nu.get((s_,t-1),0) for s_,t in zip(sub_.stay_id,sub_.hour)]
sub_['D1']=[Ptu2.D.get((s_,t-1),0) for s_,t in zip(sub_.stay_id,sub_.hour)]
sub_['G1']=[Ptu2.Gany.get((s_,t-1),np.nan) for s_,t in zip(sub_.stay_id,sub_.hour)]
grp_=sub_.stay_id.values
oh=np.eye(17)[sub_.lb.values]
for nm,F in [('q',q_[mG]),('q+lb',np.hstack([q_[mG],oh*5])),('wc',wc_[mG]),('wc+lb',np.hstack([wc_[mG],oh*5]))]:
    r1,pr1=cvr(F,sub_.nu1.values.astype(float))
    b_=np.polyfit(sub_.nu1.values,pr1,1)[0]
    print(nm,'R2 nu1',r1,'slope(pred on nu1)',round(b_,3))
print('share of finger rows where last finger reading is from t-1:', round((sub_.lastage==3600).mean(),3))
```