The bin calibration improves fitting-row loss but not both validation sets, so I’m rejecting it without changing weights. Another plausible issue is that doubling creates larger feature contrasts than halving, which can dominate the regression. I’ll rebalance those constraints analytically and look for stronger halving responses within the factual budget; GPU rollouts will still be the deciding test.

```python
from scipy.linalg import solve as solve281

print("Paired keys", list(paired214))
print("Feature keys", list(arch263["cf"]))
partial281 = {}
for key281, p281 in paired214.items():
    use281 = fitmask179[p281["ids"]]
    Fi281, Ff281 = arch263["cf"][key281]
    D281 = (Fi281-Ff281)[:,use281].reshape(-1,193)
    Z281 = (p281["target"]-p281["lf"])[:,use281].reshape(-1,17)
    Ds281 = D281/stats263["scale"]
    partial281[key281] = (
        Ds281.T@Ds281/len(D281),
        Ds281.T@centre234(Z281-D281@stats263["T0"])/len(D281)
    )

for drug281 in ("insulin", "norepinephrine"):
    keys281 = [k for k in partial281 if k[0] == drug281]
    G281 = sum(partial281[k][0] for k in keys281)/len(keys281)
    R281 = sum(partial281[k][1] for k in keys281)/len(keys281)
    print(drug281, "reconstruction errors",
          np.max(np.abs(G281-stats263["GD"][drug281])),
          np.max(np.abs(R281-stats263["RD"][drug281])))

def weighted_fit281(etaI, etaN2, half_ratio):
    GN281 = .5*(partial281[("norepinephrine",2.0)][0] +
                half_ratio*partial281[("norepinephrine",.5)][0])
    RN281 = .5*(partial281[("norepinephrine",2.0)][1] +
                half_ratio*partial281[("norepinephrine",.5)][1])
    A281 = (stats263["GF"] + etaI*stats263["GD"]["insulin"] +
            etaN2*GN281 + 1e-4*np.eye(193))
    B281 = (stats263["RF"] + etaI*stats263["RD"]["insulin"] +
            etaN2*RN281)
    return stats263["T0"] + solve281(
        A281, B281, assume_a="pos", check_finite=False
    )/stats263["scale"][:,None]

check281 = weighted_fit281(.25,.25,1.)
print("Baseline solution error",
      float(np.max(np.abs(check281-grid269[(.25,.25)]["T"]))))

weighted281 = {}
for etaI281 in (.20,.25,.30):
    for etaN281 in (.10,.15,.20,.25):
        for ratio281 in (2.,4.,8.):
            T281 = weighted_fit281(etaI281,etaN281,ratio281)
            s281 = score234(arch263,T281,stats263["Ftune"])
            slopes281 = np.asarray(s281[3])
            anchor281 = np.array([-1.95,-1.95,3.1,3.1])
            error281 = float(np.mean(((slopes281-anchor281)/anchor281)**2))
            weighted281[(etaI281,etaN281,ratio281)] = {
                "T": T281, "score": s281, "slope_error": error281
            }

print("=== BEST FACTUAL-FEASIBLE REBALANCED HEADS ===")
feasible281 = [
    (k,v) for k,v in weighted281.items()
    if max(v["score"][1:3]) < .0045
]
for k281,v281 in sorted(feasible281,key=lambda kv:kv[1]["slope_error"])[:14]:
    print(k281, "NLL", np.round(v281["score"][:3],6),
          "slopes", np.round(v281["score"][3],3),
          "proxy", round(v281["slope_error"],5))
if not feasible281:
    print("No candidates below the offline factual cutoff.")
    for k281,v281 in sorted(weighted281.items(),
                           key=lambda kv:max(kv[1]["score"][1:3]))[:6]:
        print(k281, v281["score"])

backup281 = {
    n: (load(n).clone() if hasattr(load(n),"clone") else np.array(load(n),copy=True))
    for n in edited()
}
with open("/work/notes.md", "a") as f281:
    f281.write("\n## Run 281 / factor-balanced causal regressions\n"
               "Rejected supervised bin calibration: no consistent validation gain. "
               "Reconstructed separate x2 and half-dose contrast statistics and "
               "fitted final heads analytically with extra half-dose weighting. "
               "No weights changed; weighted281 retains candidates, backup281 "
               "preserves balanced .25/.25.\n")
```