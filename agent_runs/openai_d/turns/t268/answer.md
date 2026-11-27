Preserving the probe response alone did not preserve factual fit. One likely cause is that the two-neuron gate lets a large signal activate the correction on unintended reading types. I’ll check that directly, then screen directions using all reading types and a fresh code gate. Success requires lower held-out loss without substantial correction on glucose or cuff readings.

```python
r231_stats = []
for label, cache231 in [("train", c210_train), ("tune100", c210_tune)]:
    X231 = cache231["X"].numpy()
    ev231 = cache231["meta"]["event"].to_numpy()
    A231 = X231 @ r230_common_v + r230_common_b
    B231 = X231 @ r230_old_v + r230_old_b
    for event231 in np.unique(ev231):
        m231 = ev231 == event231
        r231_stats.append(dict(split=label, event=event231,
                              signal_mean=float(B231[:,m231].mean()),
                              signal_sd=float(B231[:,m231].std()),
                              gate_median=float(np.median(A231[:,m231])),
                              positive_arm_fraction=float(np.mean(
                                  (A231[:,m231]+np.abs(B231[:,m231])) > 0))))
print("=== EXISTING NE SIGNAL/GATE ===")
print(pd.DataFrame(r231_stats).to_string(index=False))

r231_all = c210_train["X"][:, r230_fit].numpy().reshape(-1,192).astype(np.float64)
r231_events = np.tile(r230_events[r230_fit], c210_train["X"].shape[0])
r231_y = np.where(r231_events=="map_arterial",1.0,-1.0)
r231_mu = r231_all.mean(0)
r231_xc = r231_all-r231_mu
r231_C0 = r231_xc.T@r231_xc/len(r231_all)
r231_Cmap = ((r231_all-r230_mean).T@(r231_all-r230_mean))/len(r231_all)
r231_cross = r231_xc.T@(r231_y-r231_y.mean())/len(r231_all)
r231_K = r231_C0+1e-3*np.trace(r231_C0)/192*np.eye(192)
r231_gate_v = np.linalg.solve(r231_K,r231_cross)
r231_invd = np.linalg.solve(r231_K,r230_d)
r231_gate_v -= r231_invd*(r231_gate_v@r230_d)/(r231_invd@r230_d)
r231_gate_b = r231_y.mean()-r231_mu@r231_gate_v
print("=== NEW GATE CHECK ===")
print("probe sensitivity:", float(r231_gate_v@r230_d))
for event231 in np.unique(r231_events):
    pred231 = r231_all[r231_events==event231]@r231_gate_v+r231_gate_b
    print(event231, np.quantile(pred231,[0,.01,.5,.99,1]))

r231_candidates = {}
r231_screen = []
for ridge231 in [1e-6,1e-5,1e-4,1e-3]:
    v231 = np.linalg.solve(r231_Cmap+ridge231*np.trace(r231_Cmap)/192*np.eye(192),r230_d)
    v231 *= r230_target/(v231@r230_d)
    b231 = -float(r230_mean@v231)
    for amp231 in [.5,1.0]:
        w231 = {n:t.clone() for n,t in r230_base.items()}
        w231[wn][48] = torch.tensor(25*r231_gate_v+v231,dtype=torch.float32)
        w231[wn][171] = torch.tensor(25*r231_gate_v-v231,dtype=torch.float32)
        w231[bn][48] = 25*r231_gate_b+b231
        w231[bn][171] = 25*r231_gate_b-b231
        w231[on][:,[48,171]] *= amp231
        r231_candidates[(ridge231,amp231)] = w231
        lt231,_ = c210_eval(w231,c210_train)
        lv231,_ = c210_eval(w231,c210_tune)
        dl231=lt231-c210_train["loss0"]
        r231_screen.append(dict(ridge=ridge231,amp=amp231,norm=np.linalg.norm(v231),
                               signal_sd=np.sqrt(v231@r230_C@v231),
                               train_fit=float(dl231[r230_fit].mean()),
                               train_validation=float(dl231[r230_val].mean()),
                               tune100=float((lv231-c210_tune["loss0"]).mean())))
print("=== ALL-CODE COVARIANCE / NEW GATE SCREEN ===")
print(pd.DataFrame(r231_screen).to_string(index=False))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 231: all-code covariance and fresh code gate\n"
            "Arterial-only covariance screening failed; checked signal/gate distributions and fitted "
            "a training-only constrained linear code classifier whose raw-probe sensitivity is zero. "
            "Screened all-code minimum-variance NE directions. No weights installed.\n")
    f.write(pd.DataFrame(r231_screen).to_string(index=False)+"\n")
```
