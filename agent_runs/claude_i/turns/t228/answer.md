The last run confirmed that jac2.py and sim.py hold helper code from earlier sessions. sim.py contains an exact local rollout of the state update, `run_stay`, which uses the posterior-mean z. JI and JN are saved arrays of shape (150 bundles, 32 z-dims, 3). I most likely computed them as sensitivities of the downstream glucose (JI) and MAP (JN) predictions to the z posterior mean at drug bundles, over 3 horizons.

My working design is a gated "deviation" feature built from z_post's 50 dead hidden units. The feature is the decoded drug rate minus the rate expected from the context, active only at that drug's bundles. It is added to the z posterior mean along a chosen direction w, so the state update carries it forward.

Whether this is feasible depends on whether some z-directions have consistent, sizeable effects on downstream predictions. I will summarise JI and JN: the mean effect per z-dim and horizon, how consistent it is across bundles, and the natural spread of z. If a few directions have consistent same-sign effects that are large relative to the natural z spread, the plan is viable. If the effects are tiny or have inconsistent signs, I need a different injection point.

```python
JI=np.load('/work/JI.npy'); JN=np.load('/work/JN.npy')
for nm,J in [('JI',JI),('JN',JN)]:
    m=J.mean(0); s=J.std(0)+1e-12
    print(nm,'mean|J| per horizon',np.round(np.abs(J).mean((0,1)),3),' norm of mean-J per horizon',np.round(np.linalg.norm(m,axis=0),3))
    top=np.argsort(-np.linalg.norm(m,axis=1))[:6]
    for d in top: print('  dim',d,'meanJ',np.round(m[d],2),'t',np.round(m[d]/(s[d]/np.sqrt(len(J))),1))
Z=np.asarray(ZZt); sd=Z.std(0)
print('Z',Z.shape,'sd pct0/25/50/75/100',np.round(np.percentile(sd,[0,25,50,75,100]),3))
```