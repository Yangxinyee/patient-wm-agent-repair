The GPU state pass completed with the intended 11 edited tensors. I’ll measure state changes before the first positive infusion and check whether the private channels are zero on drug-free bundles, then restore the neutral edit. Large pre-treatment changes would explain the factual failure; small ones would point instead to damage when the response activates.

```python
so263 = np.load("/work/fwd_25/states_original.npz")
se263 = np.load("/work/fwd_25/states_edited.npz")
xo263, xe263 = so263["x"], se263["x"]
tt263 = pd.read_csv("/data/tuning.csv")
starts263 = tt263[(tt263.event.isin(["insulin","norepinephrine"])) & (tt263.rate > 0)]
first263 = starts263.groupby("stay_id").hour.min().to_dict()
before263 = np.array([h < first263.get(int(s), np.inf)
                     for s,h in zip(se263["stay_id"],se263["hour"])])
inactive263 = ~(se263["has_insulin"].astype(bool) | se263["has_norepinephrine"].astype(bool))
priv263 = xe263[:, [960+16, 960+12]]
outdiff263 = np.sqrt(np.mean((xe263[:,:192]-xo263[:,:192])**2,axis=1))
hdiff263 = np.sqrt(np.mean((xe263[:,192:960]-xo263[:,192:960])**2,axis=1))
for label263, mask263 in (("before_any_positive_drug",before263),
                          ("no_drug_in_bundle",inactive263),
                          ("drug_bundle",~inactive263),("all",np.ones(len(xe263),bool))):
    print(label263, "n",int(mask263.sum()),
          "out_RMS",float(np.sqrt(np.mean(outdiff263[mask263]**2))),
          "h_RMS",float(np.sqrt(np.mean(hdiff263[mask263]**2))),
          "private_RMS",np.sqrt(np.mean(priv263[mask263]**2,axis=0)).tolist(),
          "private_abs_max",np.max(np.abs(priv263[mask263]),axis=0).tolist())
tab263 = pd.DataFrame({"stay":se263["stay_id"],"hour":se263["hour"],
                      "insulin":se263["has_insulin"],"ne":se263["has_norepinephrine"],
                      "private_I":priv263[:,0],"private_N":priv263[:,1],
                      "out_RMS":outdiff263,"h_RMS":hdiff263})
print("FIRST STAY",tab263[tab263.stay==tab263.stay.iloc[0]].head(10).round(6).to_string(index=False))
print("EARLY WORST",tab263.loc[np.flatnonzero(before263)].sort_values("out_RMS",ascending=False).head(5).round(6).to_string(index=False))
print("=== RESPONSE-FIT CONTRACTS ===")
import inspect
for name263 in ("encode_actions168","candidate_readout216","readout214","replay_token214"):
    print(name263,inspect.signature(globals()[name263]))
for name263 in ("A215","dz215","zbase215","Xbase215","native215","responses215",
                "factual_means216","factual_logits216","drug_labels215","dr215",
                "official_h182","base_h214"):
    if name263 in globals():
        vv263=globals()[name263]
        print(name263, type(vv263).__name__,
              list(vv263.keys()) if isinstance(vv263,dict) else np.shape(vv263))
for name263 in list(edited()):
    drop_edit(name263)
neutral263=np.load("/work/neutral209.npz")
for name263 in neutral263.files:
    save_edit(name263,neutral263[name263].astype(np.float32))
with open("/work/notes.md","a") as f:
    f.write("\n## Forward 24; Python 263 — actual GPU state diagnostic\n"
            "- Compared sharp243 GPU states against original, including pre-treatment and drug-free bundles.\n"
            "- Restored neutral209 after diagnosis; sharp243 remains rejected.\n")
print("RESTORED",edited())
```
