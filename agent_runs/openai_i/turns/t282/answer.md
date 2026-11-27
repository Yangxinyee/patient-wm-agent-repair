The inspection stopped on another type-checking error; no weights changed. It did show that the direct norepinephrine direction is much smaller than the custom attention direction. I’ll test modest direct-token gains locally before installing anything. Useful dose sensitivity with little factual distribution drift would support this simpler approach; saturation or large drift would rule it out.

```python
import time
print("=== STARTUP COVERAGE AND INACTIVE INNOVATIONS ===")
hh251 = np.asarray(hours156)
aa251 = np.asarray(innovation237)
mm251 = np.asarray(inactive237, dtype=bool)
for label, mask in (("negative",hh251<0),("zero",hh251==0),("later",hh251>0)):
    m = mask & mm251
    print(label, "rows", int(mask.sum()), "inactive", int(m.sum()),
          "innovation_max", np.max(np.abs(aa251[m]),axis=0) if m.any() else None)
print(pd.read_csv("/data/train.csv",nrows=12)[["stay_id","hour","event","rate"]].to_string(index=False))

print("\n=== RELEVANT RETAINED VARIABLES ===")
for nm,v in list(globals().items()):
    if nm.endswith(("216","222","237","243")) and any(t in nm.lower() for t in
            ("mean","base","result","gate","axis","private","token","feature","innovation","inactive","response")):
        if isinstance(v,(np.ndarray,torch.Tensor)):
            print(nm, tuple(v.shape))
        elif isinstance(v,dict):
            print(nm, "keys", list(v)[:6])
        elif isinstance(v,(int,float)):
            print(nm,v)

print("\n=== DIRECT-TOKEN FINITE-DOSE TEST ===")
vec251 = np.column_stack([array196(mean_directions216[d])
                         for d in ("insulin","norepinephrine")])
zero251 = np.zeros((192,2),dtype=np.float32)
base251 = candidate_readout216(zero251, (1.,1.))
meanbase251, logitsbase251 = map(array196,base251)
print("baseline shapes",meanbase251.shape,logitsbase251.shape,"contexts",n182)
rates251 = np.asarray(R193)[np.asarray(ids182,dtype=int)]
from scipy.special import softmax, log_softmax
lpbase251 = log_softmax(logitsbase251,axis=-1)
pb251 = np.exp(lpbase251)
directstats251 = []
if meanbase251.shape == (len(rates251),4):
    for gain in (0.1,0.3,1.0,3.0):
        start = time.time()
        vectors = vec251 * gain
        mf,lf = map(array196,candidate_readout216(vectors,(1.,1.)))
        kl = float(np.mean(np.sum(pb251*(lpbase251-log_softmax(lf,axis=-1)),axis=-1)))
        slopes = []
        for di,scales in enumerate(((2.,1.),(1.,2.))):
            me,le = map(array196,candidate_readout216(vectors,scales))
            active = rates251[:,di]>0
            slopes.append(np.sum(me[active]-mf[active],axis=0)/np.sum(rates251[active,di]))
        row = {"gain":gain,"KL_proxy":kl,
               "factual_mean_RMS":float(np.sqrt(np.mean((mf-meanbase251)**2))),
               "insulin_slopes":np.round(slopes[0],4).tolist(),
               "NE_slopes":np.round(slopes[1],4).tolist(),
               "seconds":round(time.time()-start,2)}
        directstats251.append(row)
        print(row)

with open("/work/notes.md","a") as f:
    f.write("\n## Python 251 — finite-dose direct latent checks\n"
            "- Run250 failed in introspection only; no weights changed.\n"
            "- Testing direct private-token gains using forward-only replay and factual distribution-drift proxies.\n"
            "- These local diagnostics do not certify factual fit or rollout effects.\n")
    for row in directstats251:
        f.write(str(row)+"\n")
```