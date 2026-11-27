The glucose-only raw-input fit is more promising: one candidate strengthens insulin responses at about +0.0016 nats on tuning data and has little latent-sample variation. It also worsens norepinephrine’s glucose response, so it is not ready. I’ll strengthen the cross-drug constraints and test a sensor gate. Success requires retaining the insulin improvement while reducing that cross-effect and reliably separating glucose from MAP inputs.

```python
wm_CI153 = np.zeros((193, 193))
wm_bI153 = np.zeros(193)
wm_CN153 = np.zeros((193, 193))
wm_bN153 = np.zeros(193)
for key, cf in wm_cf146.items():
    active = wm_split151 & wm_glucose152 & np.asarray(cf["active"], dtype=bool)
    D = ((cf["inputs"].numpy()[:, active] - wm_X152[:, active]) /
         wm_s152).reshape(-1, 192)
    D = np.column_stack([D, np.zeros(len(D))])
    y = np.tile(wm_tau_targets151[key][active], cf["inputs"].shape[0])
    if key[0] == "insulin":
        wm_CI153 += D.T @ D / len(D) / 2
        wm_bI153 += D.T @ y / len(D) / 2
    else:
        wm_CN153 += D.T @ D / len(D) / 2
        wm_bN153 += D.T @ y / len(D) / 2

wm_fits153, wm_rows153 = {}, []
for cross_weight in [1.0, 4.0, 16.0]:
    for fw in [0.3, 1.0, 3.0]:
        coef = sla.solve(wm_CI153 + cross_weight*wm_CN153 + fw*wm_K152 +
                         1e-4*np.diag([1.0]*192+[0.0]),
                         wm_bI153 + cross_weight*wm_bN153, assume_a="pos")
        w = np.r_[coef[:-1]/wm_s152, coef[-1]]
        wm_fits153[(cross_weight, fw)] = w
        nll, mean, tau, _ = wm_rawloss152(wm_train145, w, wm_glucose152)
        tnll, _, ttau, _ = wm_rawloss152(wm_tune145, w, wm_tune_glucose152)
        row = dict(cross=cross_weight, fw=fw,
                   cv_dNLL=float((nll[:, ~wm_split151]-wm_train145["nll"][:, ~wm_split151]).mean()),
                   tuning_dNLL=float((tnll-wm_tune145["nll"]).mean()),
                   latent_sd=float(ttau[:, wm_tune_glucose152].std(0).mean()))
        for key, cf in wm_cf146.items():
            cmc = dict(cf, centres=wm_train145["centres"], target=wm_train145["target"])
            _, cmean, _, _ = wm_rawloss152(cmc, w, wm_glucose152)
            mask = (~wm_split151) & (wm_events151=="glucose_finger") & np.asarray(cf["active"], dtype=bool)
            dr = np.asarray(cf["delta_rate"])[mask]
            ds = (cmean-mean).numpy()[mask]
            row[("I" if key[0]=="insulin" else "N_to_G")+str(key[1])] = float(dr@ds/(dr@dr))
        wm_rows153.append(row)
wm_frontier153 = pd.DataFrame(wm_rows153)
print("=== CROSS-DRUG-CONSTRAINED FRONTIER ===")
print(wm_frontier153.to_string(index=False))

F153 = wm_Xs152.reshape(-1, 193)
yg153 = np.tile(np.where(wm_glucose152, 1.0, -1.0), wm_X152.shape[0])
wm_gate_coef153 = sla.solve(F153.T@F153/len(F153) +
                            1e-4*np.diag([1.0]*192+[0.0]),
                            F153.T@yg153/len(F153), assume_a="pos")
wm_gate153 = np.r_[wm_gate_coef153[:-1]/wm_s152, wm_gate_coef153[-1]]
print("\n=== LINEAR SENSOR-GATE MARGINS ===")
for label, mc, mask in [("train", wm_train145, wm_glucose152),
                         ("tuning", wm_tune145, wm_tune_glucose152)]:
    score = (mc["inputs"].numpy()@wm_gate153[:-1] + wm_gate153[-1])
    margin = score*np.where(mask, 1.0, -1.0)[None, :]
    print(label, "accuracy", float((margin>0).mean()),
          "margin percentiles", np.percentile(margin, [0, .1, 1, 50, 99, 100]))
for key, cf in wm_cf146.items():
    score = cf["inputs"].numpy()@wm_gate153[:-1] + wm_gate153[-1]
    margin = score*np.where(wm_glucose152, 1.0, -1.0)[None, :]
    print(key, "minimum gate margin", float(margin.min()))
print("Active edits:", edited())
```
