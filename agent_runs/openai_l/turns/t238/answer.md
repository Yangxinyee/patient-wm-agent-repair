The targets correctly impose about −1.95 glucose units per added insulin-rate unit and +3.1 MAP units per added norepinephrine-rate unit. Several hidden units also have very small removal scores. I’ll test two compact replacements using closed-form regression. They must improve dose responses while keeping held-out factual NLL within the allowance.

```python
def make_arch215(style, lam=.1):
    nrep = 2 if style == "linear2" else 4
    units = pruneorder214[:nrep].copy()
    rows = []
    for drug in ["insulin", "norepinephrine"]:
        sign = 1. if drug == "insulin" else -1.
        a = .25 if drug == "insulin" else 1.
        d = directions207[(drug, lam)].copy()
        other = ~family204[drug]
        off = -sign*gatecanon200/2
        off = off.copy()
        off[-1] += .5
        d -= np.mean(Xraw204[:, other] @ d)*off

        values = [Xraw204[:, other] @ (a*d)]
        for key, p in paired214.items():
            if key[0] == drug:
                othercase = ~family204[drug][p["ids"]]
                values.append(p["xi"][:, othercase] @ (a*d))
        extreme = max(float(np.max(np.abs(v))) for v in values)
        gate_strength = (extreme + 12.)/2
        gate = sign*gatecanon200.copy()
        gate[-1] -= 1.
        if style == "linear2":
            row = a*d + gate_strength*gate
            row[-1] += 6.
            rows.append(row)
        else:
            rows += [a*d + gate_strength*gate,
                     -a*d + gate_strength*gate]

    rows = np.stack(rows)
    F = Hf214.copy()
    F[..., units] = gelu214(Xraw204 @ rows.T)
    cf = {}
    for key, p in paired214.items():
        Fi = np.concatenate([p["hi"], np.ones((*p["hi"].shape[:2], 1))], axis=-1)
        Ff = np.concatenate([p["hf"], np.ones((*p["hf"].shape[:2], 1))], axis=-1)
        Fi[..., units] = gelu214(p["xi"] @ rows.T)
        Ff[..., units] = gelu214(p["xf"] @ rows.T)
        cf[key] = (Fi, Ff)
    return dict(style=style, lam=lam, units=units, rows=rows, F=F, cf=cf)

def fit_arch215(arch, eta, ridge):
    F = arch["F"]
    T0 = headbase214.copy()
    T0[arch["units"]] = 0.
    X = F[:, fitmask179].reshape(-1, 193)
    Y = (Lorig179-offset204)[:, fitmask179].reshape(-1, 17)
    scale = np.maximum(np.sqrt(np.mean(X**2, axis=0)), .05)
    Xs = X/scale
    GF = Xs.T@Xs / len(X)
    RF = Xs.T@(Y-X@T0) / len(X)
    GD = np.zeros((193, 193))
    RD = np.zeros((193, 17))
    count = 0
    for key, p in paired214.items():
        use = fitmask179[p["ids"]]
        Fi, Ff = arch["cf"][key]
        D = (Fi-Ff)[:, use].reshape(-1, 193)
        Z = (p["target"]-p["lf"])[:, use].reshape(-1, 17)
        Ds = D/scale
        GD += Ds.T@Ds
        RD += Ds.T@(Z-D@T0)
        count += len(D)
    GD /= count
    RD /= count
    A = np.linalg.solve(GF + eta*GD + ridge*np.eye(193), RF + eta*RD)
    T = T0 + A/scale[:, None]
    L = F@T + offset204
    factual_delta = nll179(L)-nll179(Lorig179)
    slopes = {}
    for drug in ["insulin", "norepinephrine"]:
        for factor in [.5, 2.]:
            key = (drug, factor)
            p = paired214[key]
            Fi, Ff = arch["cf"][key]
            Li = Fi@T + offset204[:, p["ids"]]
            Lf = Ff@T + offset204[:, p["ids"]]
            dm = ((probs214(Li)-probs214(Lf))*C179[p["ids"]][None]).sum(-1)
            use = p["own"] & valmask179[p["ids"]]
            rates = p["dr"][use]
            slopes[key] = np.sum(dm[:, use]*rates)/(8*np.sum(rates**2))
    return dict(arch=arch, eta=eta, ridge=ridge, T=T,
                train_delta=float(factual_delta[fitmask179].mean()),
                val_delta=float(factual_delta[valmask179].mean()),
                slopes=slopes)

arches215 = {style: make_arch215(style) for style in ["linear2", "pair4"]}
candidates215 = {}
for style, arch in arches215.items():
    print("\nARCHITECTURE:", style, "replaced units", arch["units"].tolist(),
          "row norms", np.round(np.linalg.norm(arch["rows"], axis=1), 2))
    for eta in [.1, 1., 10.]:
        result215 = fit_arch215(arch, eta, 1e-4)
        candidates215[(style, eta)] = result215
        print("eta", eta, "train/validation NLL deltas",
              round(result215["train_delta"], 6), round(result215["val_delta"], 6),
              "held-out slopes",
              {str(k): round(float(v), 3) for k, v in result215["slopes"].items()})
print("No weights changed.")
```
