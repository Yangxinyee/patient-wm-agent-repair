The existing hidden features cannot provide the needed response without large factual changes. I’ll test features before the value-head nonlinearity, with separate glucose and MAP corrections. Stronger held-out responses at low factual cost would support adding a small branch; failure would point to the recurrent state instead.

```python
from sklearn.linear_model import Ridge

r124_raw = r121_np(verified_head["inputs"]).astype(float)
r124_rm = r124_raw[r122_fit].mean(0)
r124_rs = np.maximum(r124_raw[r122_fit].std(0), 0.05)
r124_F = np.column_stack(((r124_raw-r124_rm)/r124_rs,
                         np.ones(len(r124_raw))))
r124_CF = {}
r124_SC = {}
for key, cf in r117_cf.items():
    xc = r121_np(cf["head"]["inputs"]).astype(float)
    r124_CF[key] = np.column_stack(((xc-r124_rm)/r124_rs,
                                    np.ones(len(xc))))
    Pc = cf["probs"].astype(float)
    r124_SC[key] = (Pc*r122_centres)@r122_basis - cf["mean"]*(Pc@r122_basis)

r124_family = head_meta.event.str.startswith("glucose").to_numpy()
r124_gate = Ridge(alpha=1.0).fit(
    r124_F[r122_fit, :-1], np.where(r124_family[r122_fit], 1., -1.))
r124_margin = r124_gate.predict(r124_F[:, :-1])*np.where(r124_family, 1., -1.)
print("Code-family gate held-out accuracy:",
      np.mean(r124_margin[r122_test] > 0),
      "minimum margin:", r124_margin[r122_test].min())

r124_systems = {}
for family, family_mask in [("insulin", r124_family),
                            ("norepinephrine", ~r124_family)]:
    Xparts, yparts, fparts = [], [], []
    for (drug, factor), cf in r117_cf.items():
        active = cf["active"] & family_mask
        Xc = (r124_SC[(drug,factor)][:,None]*r124_CF[(drug,factor)]
              - r122_s0[:,None]*r124_F)/r122_scale[:,None]
        beta = (-2.1 if family=="insulin" else 3.4) if drug==family else 0.
        yc = (beta*cf["delta_rate"]-(cf["mean"]-r117_base_mean))/r122_scale
        Xparts.append(Xc[active])
        yparts.append(yc[active])
        fparts.append(r122_fit[active])
    Xc, yc, fc = np.concatenate(Xparts), np.concatenate(yparts), np.concatenate(fparts)
    ff = r122_fit & family_mask
    Gc = Xc[fc].T@Xc[fc]/fc.sum()
    bc = Xc[fc].T@yc[fc]/fc.sum()
    Gf = (r124_F[ff].T*r122_Vb[ff])@r124_F[ff]/ff.sum()
    r124_systems[family] = (Gc,bc,Gf)

r124_results, r124_models = [], {}
for fw in [1e-4, 1e-3, 1e-2, 0.1]:
    for ridge in [1e-7, 1e-5, 1e-3]:
        coefs = {}
        for family, (Gc,bc,Gf) in r124_systems.items():
            penalty = np.eye(r124_F.shape[1])*ridge
            penalty[-1,-1] *= .01
            coefs[family] = solve(Gc+fw*Gf+penalty,bc,assume_a="pos")
        tau = np.where(r124_family, r124_F@coefs["insulin"],
                       r124_F@coefs["norepinephrine"])
        Pn = r122_tilt(r122_P,tau)
        dnll = np.log(np.maximum(r122_P[np.arange(len(Pn)),r122_bins],1e-30)) \
               -np.log(np.maximum(Pn[np.arange(len(Pn)),r122_bins],1e-30))
        rec = {"fw":fw,"ridge":ridge,"cv_dNLL":dnll[r122_test].mean(),
               "cv_tilt_rms":np.sqrt(np.mean(tau[r122_test]**2))}
        mun = (Pn*r122_centres).sum(1)
        for key, cf in r117_cf.items():
            tauc = np.where(r124_family,
                            r124_CF[key]@coefs["insulin"],
                            r124_CF[key]@coefs["norepinephrine"])
            shift = (r122_tilt(cf["probs"],tauc)*r122_centres).sum(1)-mun
            drug,factor=key
            sensor="glucose_finger" if drug=="insulin" else "map_arterial"
            m=cf["active"] & r122_test & (head_meta.event.to_numpy()==sensor)
            dr=cf["delta_rate"][m]
            rec[f"{drug[0]}{factor}_slope"]=np.dot(dr,shift[m])/np.dot(dr,dr)
        r124_results.append(rec)
        r124_models[(fw,ridge)]=coefs
r124_results=pd.DataFrame(r124_results)
print(r124_results.to_string(index=False,float_format=lambda x:f"{x:.6g}"))
print("Active edits:",edited())
```
