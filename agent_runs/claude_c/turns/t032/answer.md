The model has essentially no causal insulin effect in teacher-forced mode. Its next-hour predicted glucose moves +0.17 ± 0.26 mg/dL per U/h of within-tier dose noise, about 8 SE from the true −2. The observed outcome on these 100 stays gives −0.81 ± 0.78, consistent with −2 given the small sample. So the pathway from dose to glucose is weak everywhere, in factual mode as well as in rollouts (about −0.3 per U/h after the first dose).

That changes the strategy. Giving the model the correct dose response should not hurt the factual fit; it should slightly improve it, since the true dose noise explains about 2 mg/dL of glucose variance. The fix is to amplify a dose-specific pathway, not to suppress a confounded one.

Next I will run the same teacher-forced check for norepinephrine: the slope of the model's predicted next arterial MAP on the NE rate, with titration-policy controls, against the observed slope. That tells me whether NE needs the same treatment.

```python
pmm = fa[fa.event=='map_arterial'].set_index(['stay_id','h'])
Pt['pmA'] = pmm.predicted_mean.reindex(Pt.index)
Pt['pmA1'] = Pt.groupby(level=0).pmA.shift(-1)
Pt['A1'] = Pt.groupby(level=0).map_arterial.shift(-1)
Pt['NE'] = Pt.norepinephrine; Pt['NEp'] = Pt.groupby(level=0).NE.shift(1)
Pt['src'] = np.where(Pt.map_arterial.notna(),1,np.where(Pt.map_cuff.notna(),2,0))
Pt['mr'] = Pt.map_arterial.fillna(Pt.map_cuff)
q = Pt.loc[stays100]
q = q[(q.index.get_level_values('h')>=1) & q.NEp.notna() & q.NE.notna() & q.mr.notna() & (q.NEp<1.0) & q.pmA1.notna()]
X = build_ctrl_ne(q) + [q.NE.values]
nm = [f'x{i}' for i in range(len(X)-1)]+['NE']
# drop all-zero columns
keep = [i for i,c in enumerate(X) if np.abs(c).sum()>0]
X2 = [X[i] for i in keep]; nm2 = [nm[i] for i in keep]
print("n=", len(q))
print("model pred:", ols(q.pmA1.values, X2, nm2).loc['NE'].values)
print("observed :", ols(q.A1.values, X2, nm2).loc['NE'].values)
```