The cached full-strength fits exceed the factual limit, while the accepted record analyses support insulin lowering glucose and norepinephrine raising MAP. I’ll test intermediate strengths using the same closed-form regression. A candidate below roughly +0.0045 offline, with stronger dose slopes, would merit another GPU audit.

```python
import numpy as np
from scipy.linalg import solve

st242 = stats234[("pair4", 0.1)]
A242 = st242["GF"] + 0.3 * sum(st242["GD"].values())
rhs242 = st242["RF"] + 0.3 * sum(st242["RD"].values())
reconstruction242 = []
for scaled242 in [False, True]:
    for residual242 in [False, True]:
        U242 = T235 - st242["T0"] if residual242 else T235.copy()
        if scaled242:
            U242 = U242 * st242["scale"][:, None]
        E242 = A242 @ U242 - rhs242
        ridge242 = -float(np.sum(E242 * U242) / np.sum(U242 * U242))
        err242 = float(np.linalg.norm(E242 + ridge242 * U242)
                       / max(np.linalg.norm(rhs242), 1e-12))
        reconstruction242.append((err242, ridge242, scaled242, residual242))
print("Normal-equation reconstructions:", reconstruction242)
best_reconstruction242 = min(reconstruction242)
print("Current score:", score234(st242["arch"], T235, st242["Ftune"]))

grid242 = {}
err242, ridge242, scaled242, residual242 = best_reconstruction242
if err242 < 1e-5 and ridge242 >= 0:
    print("Recovered exact fitting equation; ridge =", ridge242)
    for etaI242, etaNE242 in [
        (0.4, 0.4), (0.5, 0.5), (0.6, 0.6),
        (0.5, 0.3), (0.6, 0.3), (0.8, 0.3),
        (0.3, 0.5), (0.3, 0.7), (0.4, 0.6), (0.5, 0.4)
    ]:
        a242 = st242["GF"] + etaI242 * st242["GD"]["insulin"] + etaNE242 * st242["GD"]["norepinephrine"]
        b242 = st242["RF"] + etaI242 * st242["RD"]["insulin"] + etaNE242 * st242["RD"]["norepinephrine"]
        u242 = solve(a242 + ridge242 * np.eye(193), b242, assume_a="pos")
        t242 = u242 / st242["scale"][:, None] if scaled242 else u242
        if residual242:
            t242 = t242 + st242["T0"]
        result242 = score234(st242["arch"], t242, st242["Ftune"])
        grid242[(etaI242, etaNE242)] = {"T": t242, "score": result242}
        print((etaI242, etaNE242), result242)
else:
    print("Equation not yet recovered; no candidate fitted or staged.")
print("Current edits unchanged:", edited())
```