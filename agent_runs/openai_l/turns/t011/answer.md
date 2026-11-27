The better-supported insulin discontinuities are consistent with a next-hour effect near −2 per unit/hour, while the pre-dose lab placebos are near zero. The lowest threshold also changes monitoring, so I won’t rely on it alone. I’ll now refit the dynamics with explicit timing; agreement would support cumulative rollout targets and resolve the earlier sign conflict.

```python
import time
from scipy.optimize import minimize
from scipy.special import expit

print("=== NUMERIC EMBEDDING TABLES ===")
print("numeric_note:", VOCAB.get("numeric_note"))
num = VOCAB["numeric"]
print("numeric type:", type(num).__name__)
if isinstance(num, dict):
    print("numeric keys:", list(num)[:20])
    for k,v in num.items():
        if str(k) in ["8","9","insulin","norepinephrine","rate","dose"]:
            print(k, str(v)[:2600])
elif isinstance(num,list):
    for v in num:
        if isinstance(v,dict) and (v.get("id") in [8,9] or v.get("code") in [8,9] or v.get("name") in ["insulin","norepinephrine"]):
            print(v)
print("Previously saved SSM fit types:", {k:type(v).__name__ for k,v in ssm_fits.items()})

# Measurement time indices: -1,0,...,24.
# Input u[t] affects the transition FROM that time TO the next time.
ssm_obs2 = {
    "glucose": (np.column_stack([pre["glucose_finger"],P["glucose_finger"][:,:25]]),
                np.column_stack([pre["glucose_lab"],P["glucose_lab"][:,:25]])),
    "MAP": (np.column_stack([pre["map_arterial"],P["map_arterial"][:,:25]]),
            np.column_stack([pre["map_cuff"],P["map_cuff"][:,:25]]))}
ssm_inputs2 = {
    "glucose": np.column_stack([np.zeros(len(rec_ids)),A["insulin"][:,:24],np.zeros(len(rec_ids))]),
    "MAP": np.column_stack([np.zeros(len(rec_ids)),A["norepinephrine"][:,:24],np.zeros(len(rec_ids))])}

def aligned_ssm_nll(theta, obs, inputs, get_innovations=False):
    mu,rho,beta = theta[:3]
    vb,vi,vq,v1,v2 = np.exp(2*np.asarray(theta[3:8]))
    offset = theta[8]
    y1,y2 = obs
    ns,nt = y1.shape
    state = np.full(ns,mu); base = np.full(ns,mu)
    p00 = np.full(ns,vb+vi); p01 = np.full(ns,vb); p11 = np.full(ns,vb)
    loss,count = 0.,0
    innovations = []
    for t in range(nt):
        for sensor,(yy,vr,bias) in enumerate([(y1,v1,0.),(y2,v2,offset)]):
            ok = np.isfinite(yy[:,t])
            resid = yy[ok,t]-state[ok]-bias
            variance = p00[ok]+vr
            loss += .5*np.sum(np.log(2*np.pi*variance)+resid*resid/variance)
            count += int(ok.sum())
            if get_innovations:
                innovations.append((t,sensor,ok.copy(),resid.copy(),variance.copy()))
            c00,c01 = p00[ok].copy(),p01[ok].copy()
            state[ok] += c00/variance*resid
            base[ok] += c01/variance*resid
            p00[ok] -= c00*c00/variance
            p01[ok] -= c00*c01/variance
            p11[ok] -= c01*c01/variance
        if t+1 < nt:
            state = rho*state+(1-rho)*base+beta*inputs[:,t]
            p00 = rho*rho*p00+2*rho*(1-rho)*p01+(1-rho)**2*p11+vq
            p01 = rho*p01+(1-rho)*p11
    return (loss/count,innovations) if get_innovations else loss/count

aligned_fits = {}
for label,mu,beta,sd in [
    ("glucose",145.,-2.,[35,25,5,9,3]),
    ("MAP",75.,3.,[12,7,2,3,6])]:
    theta = np.array([mu,.9,beta,*np.log(sd),0. if label=="glucose" else 3.])
    start = time.time()
    result = minimize(aligned_ssm_nll,theta,args=(ssm_obs2[label],ssm_inputs2[label]),
                      method="L-BFGS-B",
                      bounds=[(None,None),(-.99,.9999),(None,None)]+[(-4,7)]*5+[(None,None)],
                      options={"maxiter":180,"ftol":1e-10,"gtol":1e-6})
    aligned_fits[label] = result
    report = dict(zip(["baseline_mean","rho","beta","baseline_SD","initial_SD","process_SD",
                       "sensor1_SD","sensor2_SD","sensor2_offset"],
                     [*result.x[:3],*np.exp(result.x[3:8]),result.x[8]]))
    print("\nALIGNED SSM",label,"success",result.success,"seconds",round(time.time()-start,1),
          "NLL",round(result.fun,6),{k:round(float(v),5) for k,v in report.items()})

# Use exactly the stay-hours returned by the original forward request when computing targets.
print("\n=== ORIGINAL ROLLOUT ROW STRUCTURE ===")
print(baseline_rollout.head(8).to_string(index=False))
print(baseline_rollout.groupby("edit").agg(n=("shift","size"),mean_shift=("shift","mean")).round(4))
print("tuning identifiers:",tuning.stay_id.drop_duplicates().head(5).tolist())

with open("/work/notes.md","a") as f:
    f.write("\n## Run 9\nThreshold evidence: insulin at 179.5 gives beta -2.04 (cluster SE .26, width12); at219.5 -2.28 (.48). Pre-dose lab jumps near zero; first150.5 threshold changes monitoring, so do not use that finger estimate alone. Implemented independent SSM with explicit observations -1..24 and action a[h] only in transition h->h+1, to test whether the old opposite-sign estimates arose from time alignment. No edits.\n")
```