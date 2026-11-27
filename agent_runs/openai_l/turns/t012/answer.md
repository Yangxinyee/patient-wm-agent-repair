Explicit timing did not resolve the conflict: the random-baseline model still disagrees with the insulin threshold evidence. Its baseline assumptions may be inappropriate for adaptively dosed stays. I’ll fit a separate dynamical model with unrestricted stay-specific intercepts and initial states. Agreement with the thresholds would support it; continued disagreement would keep the cumulative effects uncertain.

```python
import numpy as np, time
from scipy.optimize import minimize

print("=== NUMERIC TABLE EXAMPLES ===")
print("number of entries",len(VOCAB["numeric"]))
for entry in VOCAB["numeric"][:3]:
    print(entry)
for entry in VOCAB["numeric"]:
    if isinstance(entry,dict) and any(str(v) in ["8","9","insulin","norepinephrine"] for k,v in entry.items()
                                      if k in ["code","code_id","id","name","code_name"]):
        print("DRUG NUMERIC",entry)

# Fixed-effect state-space likelihood. Each stay has its own unrestricted
# dynamical intercept c_i and initial latent level x_i. Profile both out exactly.
# No world-model weights are fitted or differentiated.
def fixed_ssm_nll(theta, obs, inputs, return_fixed=False, reml=True):
    rho,beta = theta[:2]
    vq,v1,v2 = np.exp(2*np.asarray(theta[2:5]))
    offset = theta[5]
    y1,y2 = obs
    ns,nt = y1.shape
    state = np.zeros(ns)
    c = np.zeros(ns)       # coefficient of stay-specific dynamical intercept
    d = np.ones(ns)        # coefficient of initial latent level
    variance_state = np.zeros(ns)
    g00 = np.zeros(ns); g01 = np.zeros(ns); g11 = np.zeros(ns)
    v0 = np.zeros(ns); v1score = np.zeros(ns)
    raw = np.zeros(ns); nobs = np.zeros(ns,dtype=int)
    for t in range(nt):
        for yy,vr,bias in [(y1,v1,0.),(y2,v2,offset)]:
            ok = np.isfinite(yy[:,t])
            residual = yy[ok,t]-bias-state[ok]
            variance = variance_state[ok]+vr
            w = 1/variance
            co,do = c[ok],d[ok]
            g00[ok] += co*co*w
            g01[ok] += co*do*w
            g11[ok] += do*do*w
            v0[ok] += co*residual*w
            v1score[ok] += do*residual*w
            raw[ok] += np.log(2*np.pi*variance)+residual*residual*w
            nobs[ok] += 1
            gain = variance_state[ok]*w
            state[ok] += gain*residual
            c[ok] *= 1-gain
            d[ok] *= 1-gain
            variance_state[ok] *= 1-gain
        if t+1<nt:
            state = rho*state+beta*inputs[:,t]
            c = rho*c+1
            d = rho*d
            variance_state = rho*rho*variance_state+vq
    det = np.maximum(g00*g11-g01*g01,1e-20)
    intercept = (g11*v0-g01*v1score)/det
    initial = (g00*v1score-g01*v0)/det
    profiled = raw-v0*intercept-v1score*initial
    if reml:
        profiled += np.log(det)
    loss = .5*profiled.sum()/np.maximum(1,(nobs-2).sum())
    return (loss,intercept,initial) if return_fixed else loss

fixed_fits = {}
for label,beta,sd,offset in [
    ("glucose",-2.,[5,9,3],0.),
    ("MAP",3.,[2,3,6],3.)]:
    start = time.time()
    theta = np.array([.9,beta,*np.log(sd),offset])
    result = minimize(fixed_ssm_nll,theta,args=(ssm_obs2[label],ssm_inputs2[label]),
                      method="L-BFGS-B",
                      bounds=[(-.99,.999),(None,None)]+[(-4,6)]*3+[(None,None)],
                      options={"maxiter":160,"ftol":1e-10,"gtol":1e-6})
    fixed_fits[label] = result
    loss,ci,xi = fixed_ssm_nll(result.x,ssm_obs2[label],ssm_inputs2[label],return_fixed=True)
    print("\nFIXED-EFFECT SSM",label,"success",result.success,"seconds",round(time.time()-start,1),
          "NLL",round(loss,6),
          "rho",round(result.x[0],5),"beta",round(result.x[1],5),
          "process/sensor SDs",np.exp(result.x[2:5]).round(4),
          "sensor offset",round(result.x[5],5))
    print("latent baseline quantiles",np.quantile(ci/(1-result.x[0]),[.05,.5,.95]).round(2),
          "initial quantiles",np.quantile(xi,[.05,.5,.95]).round(2))

# Determine how much dose variation is persistent between stays.
fg_last = Last["glucose_finger"][:,:24]
ins_base = np.select([fg_last>299,fg_last>259,fg_last>219,fg_last>179,fg_last>150],
                     [6.,4.,3.,2.,1.],default=np.nan)
ratio = A["insulin"][:,:24]/ins_base
ratio[A["insulin"][:,:24]<=0] = np.nan
count = np.isfinite(ratio).sum(axis=1)
mean_ratio = np.nansum(ratio,axis=1)/np.maximum(count,1)
within = np.nansum((ratio-mean_ratio[:,None])**2)/np.maximum(1,np.isfinite(ratio).sum())
print("\nInsulin dose/base ratio: overall SD",round(np.nanstd(ratio),4),
      "within-stay SD",round(np.sqrt(within),4),
      "stay-mean quantiles",np.quantile(mean_ratio[count>=6],[.1,.5,.9]).round(4))

with open("/work/notes.md","a") as f:
    f.write("\n## Run 10\nExplicitly aligned random-baseline SSM still conflicts with threshold evidence (glucose rho at .9999, beta -.033; MAP beta -.377). Therefore timing alone does not explain it. Fitting a separate profiled fixed-effect SSM with unrestricted stay-specific intercept and initial state; random-effect independence from adaptively selected treatments is suspect. This is a small record model, not world-model training.\n")
```