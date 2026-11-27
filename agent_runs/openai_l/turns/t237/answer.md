The tuple contains an intervened input and its change from factual input. With that interpretation, the dose projections are near unit scale on the intended sensor family. I’ll verify the target mean shifts and rank expendable hidden units; small removal error would support a compact nonlinear replacement.

```python
def gelu214(x):
    return x * ndtr(x)

def probs214(l):
    return np.exp(l - logsumexp(l, axis=-1, keepdims=True))

Waug214 = np.concatenate([Wbase204, bbase204[:, None]], axis=1)
headbase214 = np.concatenate([Wfinal204.T, bfinal204[None, :]], axis=0)
Hf214 = np.concatenate([H179, np.ones((*H179.shape[:2], 1))], axis=-1)
paired214 = {}
print("TARGET MEAN-SHIFT CHECK:")
for key214, tup214 in cases207.items():
    ids214, xi214, dx214, dr214 = tup214
    xf214 = xi214 - dx214
    hi214 = gelu214(xi214 @ Waug214.T)
    hf214 = gelu214(xf214 @ Waug214.T)
    li214 = hi214 @ Wfinal204.T + bfinal204 + offset204[:, ids214]
    lf214 = hf214 @ Wfinal204.T + bfinal204 + offset204[:, ids214]
    own214 = family204[key214[0]][ids214]
    mi214 = (probs214(li214) * C179[ids214][None]).sum(-1)
    mf214 = (probs214(lf214) * C179[ids214][None]).sum(-1)
    mt214 = (probs214(li214 + targets179[key214]) *
             C179[ids214][None]).sum(-1)
    slope_orig214 = np.sum((mi214-mf214)[:, own214]*dr214[own214]) / (
        8*np.sum(dr214[own214]**2))
    slope_target214 = np.sum((mt214-mf214)[:, own214]*dr214[own214]) / (
        8*np.sum(dr214[own214]**2))
    paired214[key214] = dict(ids=ids214, xi=xi214, xf=xf214, dx=dx214,
                            dr=dr214, own=own214, hi=hi214, hf=hf214,
                            li=li214, lf=lf214, target=li214+targets179[key214])
    if key214[1] in [.5, 2.]:
        print(key214, "factual alignment RMS",
              np.sqrt(np.mean((xf214-Xraw204[:, ids214])**2)),
              "original slope", round(slope_orig214, 4),
              "target slope", round(slope_target214, 4))

F214 = Hf214[:, fitmask179].reshape(-1, 193)
cov214 = F214.T @ F214 / len(F214)
reg214 = 1e-6*np.trace(cov214[:192, :192])/192
inv214 = np.linalg.inv(cov214 + reg214*np.diag([1.]*192+[0.]))
prunescore214 = np.sum(Wfinal204**2, axis=0) / np.diag(inv214)[:192]
pruneorder214 = np.argsort(prunescore214)
print("\nLOWEST REMOVAL SCORES:",
      [(int(j), round(float(prunescore214[j]), 8)) for j in pruneorder214[:16]])

dosecenter214 = {}
for drug214 in ["insulin", "norepinephrine"]:
    sign214 = 1. if drug214 == "insulin" else -1.
    own214 = family204[drug214]
    d214 = directions207[(drug214, .1)].copy()
    othermean214 = np.mean(Xraw204[:, ~own214] @ d214)
    offgate214 = -sign214*gatecanon200/2
    offgate214 = offgate214.copy()
    offgate214[-1] += .5
    d214 -= othermean214*offgate214
    dosecenter214[drug214] = d214
    print(drug214, "centered own/other projection quantiles",
          np.quantile(Xraw204[:, own214] @ d214, [.001, .5, .999]),
          np.quantile(Xraw204[:, ~own214] @ d214, [.001, .5, .999]))
print("No weights changed.")
```
