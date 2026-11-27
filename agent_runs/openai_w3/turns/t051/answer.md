Both glucose sensors give nearly identical latent directions, as do both MAP sensors. That supports using one correction direction per drug. I’ll now separate drug-start detection from dose estimation: fitting the rate only when that drug is present may avoid the large errors of the joint extractor. Low held-out rate error and few false drug detections would justify a first edit.

```python
from sklearn.linear_model import LogisticRegression

r46_U = np.asarray(r41_X) * np.asarray(r41_uscale)[None,:]
r46_train = np.asarray(r41_fitmask).astype(bool)
assert r46_train.any() and (~r46_train).any()
print("fit/test counts:", r46_train.sum(), (~r46_train).sum())
print("coefficient scaling check:",
      np.max(np.abs(r41_action_fit.coef_/r41_uscale[None,:] - r41_action_coef)))

r46_models = {}
print("\n=== CONDITIONAL ACTION EXTRACTORS ===")
for j,drug in enumerate(["insulin","norepinephrine"]):
    present = np.asarray(r41_Y)[:,j+2] > .5
    fit = r46_train & present
    test = (~r46_train) & present
    best = None
    for alpha in [.001,.01,.1,1.,10.]:
        reg = Ridge(alpha=alpha).fit(r41_X[fit], r41_Y[fit,j])
        rmse = float(np.sqrt(np.mean((reg.predict(r41_X[test])-r41_Y[test,j])**2)))
        if best is None or rmse < best[0]:
            best = (rmse,alpha)
    rate_reg = Ridge(alpha=best[1]).fit(r41_X[present], r41_Y[present,j])
    clf_cv = LogisticRegression(C=10,max_iter=500,solver="lbfgs").fit(
        r41_X[r46_train],present[r46_train])
    cp = clf_cv.predict(r41_X[~r46_train])
    true = present[~r46_train]
    print(drug, "conditional rate RMSE", best[0], "alpha",best[1],
          "presence false-positive",float(np.mean(cp[~true])),
          "false-negative",float(np.mean(~cp[true].astype(bool))))
    clf = LogisticRegression(C=10,max_iter=500,solver="lbfgs").fit(r41_X,present)
    r46_models[drug] = {
        "rate_w":rate_reg.coef_/r41_uscale, "rate_b":float(rate_reg.intercept_),
        "gate_w":clf.coef_[0]/r41_uscale, "gate_b":float(clf.intercept_[0]),
        "rate_rmse":best[0]
    }

print("\n=== CONTEXTUAL EXPECTED DOSE ===")
r46_fold = np.asarray(S["stay_id"]) % 2 == 0
for j,drug in enumerate(["insulin","norepinephrine"]):
    active = np.asarray(r41_present)[:,j] > .5
    best = None
    for alpha in [.1,1.,10.,100.,1000.]:
        reg = Ridge(alpha=alpha).fit(np.asarray(r41_ctx)[active&r46_fold],
                                     r41_rates[active&r46_fold,j])
        rmse = float(np.sqrt(np.mean((reg.predict(np.asarray(r41_ctx)[active&~r46_fold])
                                      -r41_rates[active&~r46_fold,j])**2)))
        if best is None or rmse < best[0]:
            best = (rmse,alpha)
    reg = Ridge(alpha=best[1]).fit(np.asarray(r41_ctx)[active],r41_rates[active,j])
    residual = r41_rates[active,j]-reg.predict(np.asarray(r41_ctx)[active])
    r46_models[drug].update(policy_w=reg.coef_,policy_b=float(reg.intercept_))
    print(drug, "held-out expected-dose RMSE",best[0],"alpha",best[1],
          "factual innovation RMS",float(np.sqrt(np.mean(residual**2))))

r46_vg = (r45_sensor_directions["glucose_finger"]["z"]
          +r45_sensor_directions["glucose_lab"]["z"])/2
r46_vm = (r45_sensor_directions["map_arterial"]["z"]
          +r45_sensor_directions["map_cuff"]["z"])/2
print("\nlatent-to-token direction norms:",
      float(torch.linalg.vector_norm(W["z_in.weight"]@torch.tensor(r46_vg))),
      float(torch.linalg.vector_norm(W["z_in.weight"]@torch.tensor(r46_vm))))

r46_input = torch.cat([torch.as_tensor(r41_ctx,dtype=torch.float32),
                       torch.as_tensor(e39,dtype=torch.float32),
                       torch.as_tensor(u39,dtype=torch.float32)],1)
r46_hidden = torch.nn.functional.gelu(linear(r46_input,"z_post.0"))
r46_outw = W["z_post.2.weight"]
r46_strength = torch.linalg.vector_norm(
    W["z_in.weight"]@r46_outw[:32],dim=0)
r46_score = np.asarray(torch.sqrt((r46_hidden**2).mean(0)) *
    (r46_strength + .1*torch.linalg.vector_norm(r46_outw[32:],dim=0)))
r46_units = np.argsort(r46_score)[:4]
print("four lowest-contribution posterior units:",r46_units.tolist(),
      "scores:",r46_score[r46_units].tolist())
with open("/work/notes.md","a") as f:
    f.write("\nPython run 46; forwards 3. Sensor latent writing directions agree strongly: glucose finger/lab cosine .99593, arterial/cuff .98941. Testing conditional action-rate extraction and logistic WORLD-presence gating before a four-unit posterior correction. No weights edited.\n")
```
