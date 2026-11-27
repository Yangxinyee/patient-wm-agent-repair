The local core matches official states to numerical precision, and the captured paired responses are weak: roughly −0.25 glucose units per insulin unit and +0.24 MAP units per norepinephrine unit. I’ll fit the missing responses at the value head, with a factual-change penalty and stay-held-out validation. Recovery on held-out stays—not training fit—will justify saving a candidate.

```python
import numpy as np
from scipy.special import logsumexp
from pathlib import Path

r180_effects = {
    "glucose_finger": np.array([-1.935, 0.0]),
    "glucose_lab": np.array([-2.015, 0.0]),
    "map_arterial": np.array([0.0, 3.053]),
    "map_cuff": np.array([0.0, 2.528]),
}
r180_case_test = rr33_stays[r179_indices] % 5 == 4
r180_fact_test = rr33_stays[r179_findices] % 5 == 4
r180_D, r180_y = {}, {}
for event in r49_sensors:
    unit = num_spec0[(event,"value")]["iqr"]
    original_delta = r179_headc[event]["M"].mean(1)-r179_head0[event]["M"].mean(1)
    r180_D[event] = (r179_Gc[event]-r179_G0[event])/unit
    r180_y[event] = (r180_effects[event][r179_drug]*r179_dc-original_delta)/unit

def r180_fisher(mask):
    S = np.zeros((193,193),dtype=np.float64)
    for event in r49_sensors:
        H = r179_headf[event]["H"][mask].reshape(-1,193)
        V = r179_Vf[event][mask].reshape(-1)
        S += (H.T @ (H*V[:,None]))/len(H)/4
    return S

def r180_fit(lam, all_cases=False):
    cm = np.ones(len(r179_indices),dtype=bool) if all_cases else ~r180_case_test
    fm = np.ones(len(r179_findices),dtype=bool) if all_cases else ~r180_fact_test
    D = np.concatenate([r180_D[e][cm] for e in r49_sensors])
    y = np.concatenate([r180_y[e][cm] for e in r49_sensors])
    S = r180_fisher(fm)
    scale = np.sqrt(np.maximum(np.diag(S),1e-8))
    Ds = D/scale[None,:]
    Ss = S/scale[:,None]/scale[None,:]
    A = Ds.T@Ds/len(Ds) + lam*Ss + 1e-8*np.eye(193)
    return np.linalg.solve(A,Ds.T@y/len(Ds))/scale

def r180_apply(data,w):
    H,P,C = (data[k] for k in ("H","P","C"))
    a = H@w
    L = np.log(np.maximum(P,1e-300))+a[:,:,None]*r179_tilt
    Z = logsumexp(L,axis=-1)
    Pnew = np.exp(L-Z[:,:,None])
    Mnew = (Pnew*C).sum(-1).mean(1)
    kl = Z-a*(P*r179_tilt).sum(-1)
    return Mnew,kl.mean(1)

r180_results = []
print("Cases train/test", int((~r180_case_test).sum()),int(r180_case_test.sum()))
print("lambda | factual KL(test) | RMS finger/MAP | slopes finger/lab/arterial/cuff | normalized effect RMSE")
for lam in (1e-5,3e-5,1e-4,3e-4,.001,.003,.01,.03,.1,.3,1.,3.):
    w = r180_fit(lam)
    slopes, rms, kls, errors = {},{},{},[]
    for event in r49_sensors:
        m0,_ = r180_apply(r179_head0[event],w)
        mc,_ = r180_apply(r179_headc[event],w)
        mf,kl = r180_apply(r179_headf[event],w)
        kls[event] = float(kl[r180_fact_test].mean())
        rms[event] = float(np.sqrt(np.mean(
            (mf[r180_fact_test]-r179_headf[event]["M"].mean(1)[r180_fact_test])**2)))
        d = 0 if event.startswith("glucose") else 1
        take = r180_case_test & (r179_drug==d)
        slopes[event] = float(np.dot(r179_dc[take],(mc-m0)[take])/
                              np.dot(r179_dc[take],r179_dc[take]))
        unit = num_spec0[(event,"value")]["iqr"]
        errors.extend(((mc-m0-r180_effects[event][r179_drug]*r179_dc) /
                       unit)[r180_case_test].tolist())
    result = {"lambda":lam,"w":w,"kl":float(np.mean(list(kls.values()))),
              "rms":rms,"slopes":slopes,"rmse":float(np.sqrt(np.mean(np.square(errors))))}
    r180_results.append(result)
    print(f"{lam:g} | {result['kl']:.6f} | "
          f"{rms['glucose_finger']:.3f}/{rms['map_arterial']:.3f} | "
          + "/".join(f"{slopes[e]:+.3f}" for e in r49_sensors)
          + f" | {result['rmse']:.5f}")

r180_eligible = [r for r in r180_results if r["kl"] <= .0025]
r180_best = min(r180_eligible if r180_eligible else r180_results,
                key=lambda r:r["rmse"] if r180_eligible else r["kl"])
r180_wfull = r180_fit(r180_best["lambda"],all_cases=True)
candidate180 = {
    "attr_heads.value_bin.2.weight":
        load_original("attr_heads.value_bin.2.weight")
        + torch.tensor(r179_tilt[:,None]*r180_wfull[None,:192],dtype=torch.float32),
    "attr_heads.value_bin.2.bias":
        load_original("attr_heads.value_bin.2.bias")
        + torch.tensor(r179_tilt*r180_wfull[192],dtype=torch.float32)
}
print("Selected held-out result:", {k:v for k,v in r180_best.items() if k!="w"})
print("Candidate max weight delta:",
      float((candidate180["attr_heads.value_bin.2.weight"]-
             load_original("attr_heads.value_bin.2.weight")).abs().max()))
print("Candidate retained but NOT saved; current edits:",edited())

with open("/work/notes.md","a") as f:
    f.write("\n## Run 180: closed-form output-layer correction\n"
            "- Fitted missing record-based immediate effects jointly for all four sensors; "
            "cross-drug effects constrained toward zero.\n"
            "- Used original-head factual Fisher penalty, ridge regularization, and "
            "stay-held-out nonlinear head replay, not model gradients.\n"
            "- Selected diagnostics: "+str({k:v for k,v in r180_best.items() if k!="w"})+"\n"
            "- candidate180 retained, not saved; absolute local-head NLL remains unvalidated.\n"
            "- Python runs: 180; forward passes: 10.\n")
```
