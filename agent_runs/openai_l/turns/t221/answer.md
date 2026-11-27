The diagnostic suggests insulin can be strengthened cheaply, while a full norepinephrine correction may exceed the factual-loss budget. Before accepting that limit, I’ll rebuild both decoders from explicitly verified head inputs, avoiding any ambiguity in the cached feature convention. A lower factual residual would make the nonlinear repair more viable; similar error would confirm the constraint.
```python
X200 = np.concatenate([arr192(I192),
                       np.ones((*I192.shape[:2],1))],axis=-1)
print("=== FEATURE CONVENTION AUDIT ===")
print("XF192 versus raw input", float(np.max(np.abs(XF192-X200))))
pre200 = I192 @ W0192.T + b0192
print("XF192 versus first-layer preactivation",
      float(np.max(np.abs(XF192[...,:192]-pre200))))
for drug in ["insulin","norepinephrine"]:
    c = cf_features81[(drug,2.0)]
    hp = torch.nn.functional.gelu(torch.tensor(
        arr192(c["input"]) @ W0192.T+b0192,dtype=torch.float32)).numpy()
    print(drug,"case hidden reconstruction max error",
          float(np.max(np.abs(hp-arr192(c["hidden"])))))

gatecanon200 = np.linalg.solve(
    X200[:,fitmask179].reshape(-1,193).T @
    X200[:,fitmask179].reshape(-1,193) /
    X200[:,fitmask179].reshape(-1,193).shape[0] + 1e-6*np.eye(193),
    X200[:,fitmask179].reshape(-1,193).T @
    gate_y192[:,fitmask179].reshape(-1) /
    X200[:,fitmask179].reshape(-1,193).shape[0])
print("canonical gate held-out RMS",
      float(np.sqrt(np.mean((X200[:,~fitmask179]@gatecanon200 -
                             gate_y192[:,~fitmask179])**2))))

directions200 = {}
canon_cases200 = {}
for drug in ["insulin","norepinephrine"]:
    fam = np.isin(events179,["glucose_finger","glucose_lab"] if drug=="insulin"
                  else ["map_arterial","map_cuff"])
    ff = X200[:,fitmask179 & fam].reshape(-1,193)
    cf = ff.T@ff/len(ff)
    cd = np.zeros((193,193))
    rd = np.zeros(193)
    for fac in [0.,.5,1.5,2.,3.]:
        c = cf_features81[(drug,fac)]
        b = cf_features81[(drug,1.)]
        rr = np.asarray(c["rows"],dtype=int)
        xc = np.concatenate([arr192(c["input"]),
                            np.ones((*c["input"].shape[:2],1))],axis=-1)
        xb = np.concatenate([arr192(b["input"]),
                            np.ones((*b["input"].shape[:2],1))],axis=-1)
        canon_cases200[(drug,fac)] = (rr,xc,xb,np.asarray(c["delta_rate"],float))
        take = fam[rr] & fitmask179[rr]
        dd = ((xc-xb)[:,take]/(fac-1)).reshape(-1,193)
        yy = np.broadcast_to(np.asarray(c["delta_rate"])[None,take]/(fac-1),
                             (8,int(take.sum()))).reshape(-1)
        cd += dd.T@dd/len(dd)/5
        rd += dd.T@yy/len(dd)/5
    eps = 1e-7*max(float(np.trace(cd)/193),1e-4)
    print("\n",drug)
    for lam in [.001,.003,.01,.03,.1,.3,1.]:
        w = np.linalg.solve(cd+lam*cf+eps*np.eye(193),rd)
        directions200[(drug,lam)] = w
        rv = (X200@w).mean(axis=0)
        hold = fam & ~fitmask179
        slope = {}
        for fac in [0.,.5,2.,3.]:
            rr,xc,xb,dr = canon_cases200[(drug,fac)]
            t = fam[rr] & ~fitmask179[rr]
            dp = ((xc-xb)@w).mean(axis=0)
            slope[fac] = round(float(np.dot(dp[t],dr[t])/
                                    np.dot(dr[t],dr[t])),3)
        print(lam,"held-out factual SD",
              round(float(rv[hold].std()),4),
              "mean",round(float(rv[hold].mean()),4),
              "physical contrast slopes",slope)
with open("/work/notes.md","a") as f:
    f.write("\n## Runs 194–200 / nonlinear-head recovery\n"
            "Recovered architecture and current quiet-wire feature caches. "
            "Sensor-family gate is nearly exact. Audited raw dose units: contrasts "
            "are correct physical rates; targets179 are logit corrections, not full "
            "logits (the apparent huge target means in run 193 were a misinterpretation). "
            "Rebuilt canonical dose directions directly from verified head inputs. "
            "No model weights changed during recovery.\n")
print("All model weights unchanged.")
```
