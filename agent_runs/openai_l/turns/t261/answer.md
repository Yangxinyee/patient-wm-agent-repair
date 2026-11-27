The fitter uses full counterfactual target logits, so the target-offset interpretation is correct. I’ll refit both the two-unit and gated four-unit designs with the independent factual cache included. This will test whether broader factual anchoring can retain useful dose sensitivity; if every stronger response still exceeds the loss limit, the current feature construction is inadequate.

```python
import scipy.linalg as sla
from scipy.special import softmax
import time
t234 = time.time()

def centre234(y):
    return y - y.mean(axis=-1, keepdims=True)

Fquiet234 = np.concatenate(
    [np.asarray(H225, dtype=np.float64),
     np.ones((8, N225, 1), dtype=np.float64)], axis=-1)
offset234 = np.asarray(Lquiet225, dtype=np.float64) - Fquiet234 @ headbase214
base_nll_train234 = nll179(Lorig179)
base_nll_tune234 = nll228(Loriginal225, B225)
expanded234 = {}
stats234 = {}

def score234(arch, T, Ftune):
    dt = nll179(arch["F"] @ T + offset204) - base_nll_train234
    dq = nll228(Ftune @ T + offset234, B225) - base_nll_tune234
    slopes = []
    for drug, factor in [("insulin", 2.), ("insulin", .5),
                         ("norepinephrine", 2.), ("norepinephrine", .5)]:
        p = paired214[(drug, factor)]
        Fi, Ff = arch["cf"][(drug, factor)]
        off = offset204[:, p["ids"]]
        shift = ((softmax(Fi @ T + off, axis=-1) -
                  softmax(Ff @ T + off, axis=-1)) *
                 C179[p["ids"]][None]).sum(-1).mean(0)
        use = p["own"] & valmask179[p["ids"]]
        dr = p["dr"][use]
        slopes.append(float(dr @ shift[use] / max(dr @ dr, 1e-12)))
    return float(dt[fitmask179].mean()), float(dt[valmask179].mean()), \
           float(dq.mean()), slopes

for style, lam in [("linear2", .1), ("pair4", .1), ("pair4", .03)]:
    arch = make_arch215(style, lam)
    rows = np.asarray(arch["rows"], dtype=np.float64)
    assert rows.shape == (len(arch["units"]), 193), rows.shape
    Ftune = Fquiet234.copy()
    Ftune[..., arch["units"]] = gelu214(np.asarray(X225, dtype=np.float64) @ rows.T)
    T0 = headbase214.copy()
    T0[arch["units"]] = 0.
    Xt = arch["F"][:, fitmask179].reshape(-1, 193)
    Xq = Ftune.reshape(-1, 193)
    Yt = (Lorig179 - offset204)[:, fitmask179].reshape(-1, 17)
    Yq = (np.asarray(Loriginal225) - offset234).reshape(-1, 17)
    scale = np.maximum(np.sqrt(.5 * (np.mean(Xt**2, axis=0) +
                                      np.mean(Xq**2, axis=0))), .05)
    Xts, Xqs = Xt / scale, Xq / scale
    GF = .5 * (Xts.T @ Xts / len(Xt) + Xqs.T @ Xqs / len(Xq))
    RF = .5 * (Xts.T @ centre234(Yt - Xt @ T0) / len(Xt) +
               Xqs.T @ centre234(Yq - Xq @ T0) / len(Xq))
    GD, RD, counts = {}, {}, {}
    for drug in ["insulin", "norepinephrine"]:
        gd, rd, count = np.zeros((193,193)), np.zeros((193,17)), 0
        for key, p in paired214.items():
            if key[0] != drug:
                continue
            use = fitmask179[p["ids"]]
            Fi, Ff = arch["cf"][key]
            D = (Fi-Ff)[:,use].reshape(-1,193)
            Z = (p["target"]-p["lf"])[:,use].reshape(-1,17)
            Ds = D / scale
            gd += Ds.T @ Ds
            rd += Ds.T @ centre234(Z-D@T0)
            count += len(D)
        GD[drug], RD[drug], counts[drug] = gd/count, rd/count, count
    stats234[(style,lam)] = dict(arch=arch, Ftune=Ftune, T0=T0,
                                scale=scale, GF=GF, RF=RF, GD=GD, RD=RD)
    for etaI, etaN in [(0.,0.), (.3,.3), (1.,1.), (1.,3.), (3.,3.)]:
        eta = {"insulin":etaI, "norepinephrine":etaN}
        M = GF.copy() + 1e-4*np.eye(193)
        R = RF.copy()
        for drug in eta:
            M += eta[drug] * GD[drug]
            R += eta[drug] * RD[drug]
        T = T0 + sla.solve(M, R, assume_a="pos") / scale[:,None]
        tr, va, tq, slopes = score234(arch,T,Ftune)
        key = (style,lam,etaI,etaN)
        expanded234[key] = dict(arch=arch,T=T,Ftune=Ftune,
                                train_delta=tr,val_delta=va,
                                tune_delta=tq,slopes=slopes)
        print(key, "NLL", np.round([tr,va,tq],6),
              "slopes I2/Ihalf/N2/Nhalf", np.round(slopes,3))
print("elapsed", round(time.time()-t234,1), "seconds")
with open("/work/notes.md","a") as f:
    f.write("\n## Run 234 / broader factual anchoring\n"
            "Refit two-unit and gated four-unit analytic value heads using "
            "the original training cache plus independent tuning factual states. "
            "No weights saved; comparing factual NLL and paired immediate-dose slopes.\n")
```
