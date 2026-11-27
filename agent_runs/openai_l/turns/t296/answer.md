The broader decoders reduce insulin’s independent factual standard deviation from 2.79 to about 0.82 and improve held-out dose-contrast accuracy for both drugs. That supports replacing the old decoder directions. I’ll fit value heads around them and first verify that the rebuilt fitting procedure reproduces the existing regression exactly.

```python
def build_stats263(arch):
    F = arch["F"]
    Ft = Fquiet234.copy()
    Ft[..., arch["units"]] = gelu214(X225 @ arch["rows"].T)
    T0 = headbase214.copy()
    T0[arch["units"]] = 0.0
    X = F[:, fitmask179].reshape(-1, 193)
    Q = Ft.reshape(-1, 193)
    scale = np.maximum(np.sqrt(0.5*(np.mean(X**2, axis=0)
                                    +np.mean(Q**2, axis=0))), 0.05)
    Xs, Qs = X/scale, Q/scale
    Y = centre234((Lorig179-offset204-F@T0)[:, fitmask179].reshape(-1, 17))
    Yq = centre234((Loriginal225-offset234-Ft@T0).reshape(-1, 17))
    GF = 0.5*(Xs.T@Xs/len(Xs)+Qs.T@Qs/len(Qs))
    RF = 0.5*(Xs.T@Y/len(Xs)+Qs.T@Yq/len(Qs))
    GD, RD = {}, {}
    for drug in ("insulin", "norepinephrine"):
        gd, rd, count = np.zeros((193,193)), np.zeros((193,17)), 0
        for key, p in paired214.items():
            if key[0] != drug:
                continue
            use = fitmask179[p["ids"]]
            Fi, Ff = arch["cf"][key]
            D = (Fi-Ff)[:, use].reshape(-1,193)
            Z = (p["target"]-p["lf"])[:, use].reshape(-1,17)
            Ds = D/scale
            gd += Ds.T@Ds
            rd += Ds.T@centre234(Z-D@T0)
            count += len(D)
        GD[drug], RD[drug] = gd/count, rd/count
    return dict(arch=arch, Ftune=Ft, T0=T0, scale=scale,
                GF=GF, RF=RF, GD=GD, RD=RD)

rebuilt263 = build_stats263(st242["arch"])
errors263 = {}
for k263 in ("scale", "GF", "RF", "T0"):
    errors263[k263] = float(np.max(np.abs(rebuilt263[k263]-st242[k263])))
for k263 in ("GD", "RD"):
    for d263 in ("insulin", "norepinephrine"):
        errors263[(k263,d263)] = float(np.max(np.abs(rebuilt263[k263][d263]-st242[k263][d263])))
print("Existing regression reconstruction:", errors263)

trial263 = {}
if max(errors263.values()) < 1e-6:
    for drug263 in ("insulin", "norepinephrine"):
        directions207[(drug263, 262.0)] = decoder262[(drug263, 1.0, 1e-8)]["d"].copy()
    arch263 = make_arch215("pair4", 262.0)
    stats263 = build_stats263(arch263)
    for ei263, en263 in ((0.3,0.3), (0.5,0.4), (0.5,0.7), (0.7,0.7),
                         (1.0,1.0), (1.5,1.5), (2.0,2.0), (1.0,2.0)):
        aa263 = (stats263["GF"] + ei263*stats263["GD"]["insulin"]
                 +en263*stats263["GD"]["norepinephrine"]+1e-4*np.eye(193))
        rr263 = (stats263["RF"] + ei263*stats263["RD"]["insulin"]
                 +en263*stats263["RD"]["norepinephrine"])
        tt263 = stats263["T0"]+np.linalg.solve(aa263,rr263)/stats263["scale"][:,None]
        sc263 = score234(arch263,tt263,stats263["Ftune"])
        trial263[(ei263,en263)] = {"T":tt263,"score":sc263}
        print((ei263,en263), "NLL", np.round(sc263[:3],6),
              "slopes", np.round(sc263[3],3))
else:
    print("No new head fitted: resolve reconstruction discrepancy first.")
```
