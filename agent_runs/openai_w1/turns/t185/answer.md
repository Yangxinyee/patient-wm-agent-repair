Rebuilding the context reduced posterior error from 0.218 to about 0.019, suggesting the stale context was the main problem. I’ll check the corrected encoder cache and estimate how glucose and MAP observations move the latent state. Stable observation-derived directions would provide a record-based target for a dose-innovation edit.

```python
print("=== CORRECTED ENCODER CACHE ===")
print("best cached encoding/config:",rr_best4)
r168_in=torch.as_tensor(rr_input4,dtype=torch.float32)
r168_hidden=TF.gelu(r168_in@ds_w0.T+ds_b0)
r168_z=r168_hidden@ds_w2[:32].T+ds_b2[:32]
r168_valid=np.asarray(rr_valid4).astype(bool)
print("posterior RMSE all/valid:",
      float(torch.sqrt(torch.mean((r168_z-r165_zref)**2))),
      float(torch.sqrt(torch.mean((r168_z[r168_valid]-r165_zref[r168_valid])**2))))
print("input differences from stale cache:",
      [float(torch.sqrt(torch.mean((r168_in[:,j:j+192]-r165_X[:,j:j+192])**2)))
       for j in [0,192,384]])
print("atom metadata examples:",[repr(q)[:220] for q in rr_desc4[:5]])
print("atom row examples:",[repr(q)[:220] for q in rr_atom_rows4[:3]])

from sklearn.linear_model import Ridge
r168_lookup={(int(s),int(round(float(h)*60))):i
             for i,(s,h) in enumerate(zip(r164_s0["stay_id"],r164_s0["hour"]))}
r168_values=np.full((len(r168_in),4),np.nan)
r168_events=["glucose_finger","glucose_lab","map_arterial","map_cuff"]
for row in r164_records[r164_records.event.isin(r168_events)].itertuples():
    ix=r168_lookup.get((int(row.stay_id),int(round(row.hour*60))))
    if ix is not None:
        r168_values[ix,r168_events.index(row.event)]=row.value
r168_physdirections={}
r168_ctx=np.asarray(r168_in[:,:192])
r168_zactual=np.asarray(r165_zref)
r168_fold={int(s):j%5 for j,s in enumerate(r164_ids)}
r168_folds=np.array([r168_fold[int(s)] for s in r164_s0["stay_id"]])
for event in ["glucose_finger","map_arterial"]:
    j=r168_events.index(event)
    mask=np.isfinite(r168_values[:,j]) & r168_valid
    y=r168_values[mask,j]
    ctx=r168_ctx[mask]
    z=r168_zactual[mask]
    folds=r168_folds[mask]
    yr=np.zeros(len(y)); zr=np.zeros_like(z)
    for f in range(5):
        tr=folds!=f; te=~tr
        reg=Ridge(alpha=30.).fit(ctx[tr],np.column_stack([y[tr],z[tr]]))
        pred=reg.predict(ctx[te])
        yr[te]=y[te]-pred[:,0]
        zr[te]=z[te]-pred[:,1:]
    direction=(yr[:,None]*zr).sum(0)/(yr@yr)
    r168_physdirections[event]=direction
    print(event,"n",len(y),"reading innovation SD",round(float(yr.std()),4),
          "latent direction/unit norm",round(float(np.linalg.norm(direction)),6),
          "latent residual explained fraction",
          round(float(np.sum((yr[:,None]*direction)**2)/np.sum(zr*zr)),4))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 168\nFresh context reduced posterior RMSE to .0186; examining earlier corrected rr_input4 cache. Estimated stay-cross-fitted latent observation directions per physical glucose/MAP unit, conditional on pre-bundle context. These directions are candidate pathways, not proof that injecting them will reproduce physiology. Counts: 168 Python runs, 11 forwards.\n")
```