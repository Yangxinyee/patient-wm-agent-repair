The baseline factual NLL is 1.63164 nats per reading on this batch. The record regressions now support responses near −2 and +3, but the rollout targets also depend on how long those responses persist. I’ll fit a small two-sensor state-space model, separate from the world model, to estimate persistence and check whether sensor noise changes that conclusion.

```python
RUNS += 1
FORWARDS = 2
from scipy.optimize import minimize
import time

for e in events[:4]:
    print(e, "quantiles:", np.round(np.nanquantile(
        A[e],[0,.001,.01,.5,.99,.999,1]),2))
print("\nOriginal shift by hour:")
print(baseline_rollout[baseline_rollout.reached].pivot_table(
    index="hour",columns="edit",values="shift",aggfunc="mean").round(3).to_string())

def kalman_nll(theta, data, drug, return_filter=False):
    mu, rho, beta = theta[:3]
    sd_b, sd_init, sd_q, sd_1, sd_2 = np.exp(theta[3:8])
    offset = theta[8]
    y1,y2 = data
    ns,nt = y1.shape
    s = np.full(ns,mu); b = np.full(ns,mu)
    p00 = np.full(ns,sd_b**2+sd_init**2)
    p01 = np.full(ns,sd_b**2); p11 = np.full(ns,sd_b**2)
    ll = 0.; count = 0
    filtered = []
    for t in range(nt):
        for y, r, bias in [(y1,sd_1**2,0.),(y2,sd_2**2,offset)]:
            ok = np.isfinite(y[:,t])
            innovation = np.where(ok,y[:,t]-bias-s,0.)
            variance = p00+r
            ll += .5*np.sum((np.log(2*np.pi*variance)+innovation**2/variance)[ok])
            count += int(ok.sum())
            k0 = np.where(ok,p00/variance,0.)
            k1 = np.where(ok,p01/variance,0.)
            old00,old01 = p00,p01
            s = s+k0*innovation; b = b+k1*innovation
            p00 = old00-k0*old00
            p01 = old01-k0*old01
            p11 = p11-k1*old01
        if return_filter:
            filtered.append(s.copy())
        if t+1<nt:
            s = rho*s+(1-rho)*b+beta*drug[:,t]
            p00 = rho*rho*p00+2*rho*(1-rho)*p01+(1-rho)**2*p11+sd_q**2
            p01 = rho*p01+(1-rho)*p11
    if return_filter:
        return ll/count,np.column_stack(filtered)
    return ll/count

ssm_fits = {}
for label,e1,e2,drug,init in [
    ("glucose","glucose_finger","glucose_lab","insulin",
     [170,.90,-2,30,15,6,8,4,0]),
    ("MAP","map_arterial","map_cuff","norepinephrine",
     [80,.90,3,12,8,2,3,6,3])
]:
    theta0 = np.array(init,dtype=float)
    theta0[3:8] = np.log(theta0[3:8])
    bounds = [
        (60,300) if label=="glucose" else (30,130),
        (.2,.995),(-12,12),
        (np.log(.2),np.log(100)),(np.log(.2),np.log(100)),
        (np.log(.2),np.log(30)),(np.log(.3),np.log(30)),
        (np.log(.3),np.log(30)),(-15,15)
    ]
    started = time.time()
    result = minimize(kalman_nll,theta0,args=((A[e1],A[e2]),A[drug]),
                      method="L-BFGS-B",bounds=bounds,
                      options={"maxiter":90,"ftol":1e-10,"maxls":25})
    ssm_fits[label] = result
    pars = result.x.copy(); pars[3:8] = np.exp(pars[3:8])
    print("\nSSM",label,"success",result.success,"seconds",round(time.time()-started,1),
          "NLL",round(result.fun,5))
    print(dict(zip(["baseline_mean","rho","drug_beta","baseline_SD","initial_SD",
                    "process_SD","sensor1_SD","sensor2_SD","sensor2_offset"],
                   np.round(pars,5))))

# Translate fitted dynamics to the same stay-hours as the baseline readout.
tune_sub = tuning[tuning.event.isin(events)].copy()
tune_sub["t"] = np.floor(tune_sub.hour+1e-5).astype(int)
tune_ids = np.sort(tuning.stay_id.unique())
tune_idx = pd.MultiIndex.from_product([tune_ids,hours],names=["stay_id","t"])
tune_rates = tune_sub.pivot(index=["stay_id","t"],columns="event",
                            values="rate").reindex(tune_idx).fillna(0)
TA = {e:tune_rates[e].to_numpy().reshape(len(tune_ids),26) for e in events[4:]}
tune_pos = {int(s):j for j,s in enumerate(tune_ids)}
linear_targets = {}
for edit,drug,label,mult in [
    ("insulin_x2","insulin","glucose",2),
    ("insulin_half","insulin","glucose",.5),
    ("ne_x2","norepinephrine","MAP",2),
    ("ne_half","norepinephrine","MAP",.5)
]:
    rho,beta = ssm_fits[label].x[1:3]
    delta = np.zeros((len(tune_ids),26))
    for t in range(12,24):
        delta[:,t+2] = rho*delta[:,t+1]+beta*(mult-1)*TA[drug][:,t+1]
    rows = baseline_rollout[(baseline_rollout.edit==edit)&baseline_rollout.reached]
    predicted = np.array([delta[tune_pos[int(s)],int(h)+1]
                          for s,h in zip(rows.stay_id,rows.hour)])
    linear_targets[edit] = float(predicted.mean())
    print("Record-based linear-dynamics target",edit,round(predicted.mean(),4))

print("\nCore/head tensors needed for a targeted edit:")
for name in NAMES:
    if ("z_prior" in name or "z_post" in name or name.startswith(
        ("gap.","core.prelude.","core.out_norm.","out_norm.","time_head.",
         "core.z_in.","core.flow.","core.coda."))):
        print(name,SHAPES[name])
with open("/work/notes.md","a") as f:
    f.write("\n## Strengthened record evidence (run 4)\n"
            "Nonlinear history controls restricted to active infusions give insulin "
            "next-hour coefficients around -2.0 (finger and lab agree); NE around "
            "+3.1 on arterial and +3.6 on cuff when both current sensors observed. "
            "Earlier near-zero estimates were misspecified, not evidence of absent effects. "
            "Original factual tuning first100 NLL = 1.63164, n=4751. "
            "Run 5 fits a separate two-sensor random-baseline state-space model "
            "to estimate persistence; this does not modify world-model weights.\n")
    for label,result in ssm_fits.items():
        f.write(f"{label} SSM parameters {result.x.tolist()}, success {result.success}.\n")
    f.write("Provisional linear rollout targets: "+json.dumps(linear_targets)+"\n")
print("Counts:",RUNS,"Python;",FORWARDS,"forward.")
```