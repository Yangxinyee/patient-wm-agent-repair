The direct prelude reconstruction matches exactly, including the synthetic dose inputs. I’ll fit small linear dose-residual models on these features, holding out whole stays. A low factual residual with strong held-out dose contrast would justify a targeted edit; poor separation would mean this pathway is also unsafe.

```python
from scipy.linalg import solve as scipy_solve
prelude_fits112 = {}
prelude_reports112 = []
for family112, factual112, counter112 in [
    ("ffn", prelude_hidden111, prelude_cf_hidden111),
    ("E", prelude_E111, prelude_cf_E111)
]:
    mu112 = factual112[residual_train102].reshape(-1, factual112.shape[-1]).mean(0).astype(np.float64)
    sd112 = factual112[residual_train102].reshape(-1, factual112.shape[-1]).std(0).astype(np.float64)
    sd112 = np.maximum(sd112, 0.02)
    X112 = np.concatenate([
        np.ones((*factual112.shape[:2],1)),
        (factual112.astype(np.float64)-mu112)/sd112
    ], axis=2)
    Xt112 = X112[residual_train102].reshape(-1, X112.shape[-1])
    Cf112 = Xt112.T @ Xt112 / len(Xt112)
    Cp112 = np.zeros_like(Cf112)
    Rp112 = np.zeros((X112.shape[-1],2))
    D112 = {}
    for key112, cf112 in counter112.items():
        dx112 = np.concatenate([
            np.zeros((*cf112.shape[:2],1)),
            (cf112.astype(np.float64)-factual112)/sd112
        ], axis=2)
        D112[key112] = dx112
        dxt112 = dx112[residual_train102].reshape(-1, X112.shape[-1])
        target112 = np.repeat(residual_targets102[key112][residual_train102],4,axis=0)
        Cp112 += dxt112.T @ dxt112 / (4*len(dxt112))
        Rp112 += dxt112.T @ target112 / (4*len(dxt112))
    ridge112 = np.eye(X112.shape[-1])*1e-4
    ridge112[0,0] = 1e-8
    family_fits112 = {}
    for gamma112 in [0.03,0.3,3.0,30.0]:
        B112 = scipy_solve(Cf112+gamma112*Cp112+ridge112,
                          gamma112*Rp112, assume_a="pos")
        f112 = X112[residual_test102] @ B112
        report112 = {"family": family112, "probe_weight": gamma112}
        for drugnum112, drug112 in enumerate(["insulin","norepinephrine"]):
            rms112 = float(np.sqrt(np.mean(f112[:,:,drugnum112]**2)))
            gains112 = {}
            rel_errors112 = {}
            for factor112 in [0.5,2.0]:
                key112 = (drug112,factor112)
                pp112 = (D112[key112][residual_test102] @ B112)[:,:,drugnum112]
                yy112 = residual_targets102[key112][residual_test102,drugnum112]
                active112 = np.abs(yy112)>1e-7
                yy_rep112 = np.repeat(yy112[active112,None],4,axis=1)
                gain112 = float(np.sum(pp112[active112]*yy_rep112)/np.sum(yy_rep112**2))
                gains112[factor112] = gain112
                rel_errors112[factor112] = float(np.sqrt(np.mean((pp112[active112]-gain112*yy_rep112)**2))/
                                                max(1e-8,np.sqrt(np.mean(yy_rep112**2))))
            avg_gain112 = (gains112[0.5]+gains112[2.0])/2
            report112[drug112+"_rms"] = rms112
            report112[drug112+"_half_gain"] = gains112[0.5]
            report112[drug112+"_x2_gain"] = gains112[2.0]
            report112[drug112+"_rms_per_gain"] = rms112/max(1e-8,abs(avg_gain112))
        prelude_reports112.append(report112)
        family_fits112[gamma112] = {"B":B112, "mu":mu112, "sd":sd112,
                                   "report":report112}
    prelude_fits112[family112] = family_fits112
    del X112, Xt112, D112
print(pd.DataFrame(prelude_reports112).round(5).to_string(index=False))
print("\nSaved-state split:",
      "training stays", len(set(sm_repair.stay_id)&set(train.stay_id)),
      "tuning stays", len(set(sm_repair.stay_id)&set(tuning.stay_id)))
print("\nState scale:",
      "h RMS", float(h109.square().mean().sqrt()),
      "out RMS", float(out109.square().mean().sqrt()),
      "E RMS", float(test_E111.square().mean().sqrt()))
```
