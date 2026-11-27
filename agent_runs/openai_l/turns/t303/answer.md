The previous validated edit is restored. I’ve also reproduced the new head’s regression exactly, so weaker versions can be computed without gradients. I’ll compare several settings for factual-loss margin and dose sensitivity; a useful setting should keep the new response gains without sitting at the loss threshold.

```python
import numpy as np
from scipy.linalg import solve
from pathlib import Path

def solve269(stats, etaI, etaNE, ridge=1e-4):
    A = (stats["GF"] + etaI*stats["GD"]["insulin"]
         + etaNE*stats["GD"]["norepinephrine"]
         + ridge*np.eye(193))
    R = (stats["RF"] + etaI*stats["RD"]["insulin"]
         + etaNE*stats["RD"]["norepinephrine"])
    return stats["T0"] + solve(A, R, assume_a="pos")/stats["scale"][:, None]

assert np.max(np.abs(
    solve269(stats263, .3, .3)-trial263[(.3,.3)]["T"]
)) < 1e-7

grid269 = {}
settings269 = [
    (.05,.05), (.1,.1), (.15,.15), (.2,.2), (.25,.25),
    (.15,.25), (.15,.35), (.2,.3), (.25,.15), (.3,.15),
    (.35,.1), (.1,.4)
]
for ei269, en269 in settings269:
    T269 = solve269(stats263, ei269, en269)
    sc269 = score234(arch263, T269, stats263["Ftune"])
    grid269[(ei269,en269)] = {"T": T269, "score": sc269}
    print((ei269,en269), "NLL", np.round(sc269[:3],6),
          "slopes", np.round(sc269[3],3))

print("build_stats263 names:", build_stats263.__code__.co_names)
print("build_stats263 locals:", build_stats263.__code__.co_varnames)
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 269 / weaker broadened-head grid\n"
            "Reproduced run263 solution to <1e-7: T=T0+solve("
            "GF+etaI*GDI+etaNE*GDN+1e-4 I, "
            "RF+etaI*RDI+etaNE*RDN)/feature_scale. "
            "Computed weaker candidates and offline scores, without saving any. "
            "Validated old .5/.4 head remains current.\n")
```