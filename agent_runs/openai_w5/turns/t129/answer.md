The prepared decoder’s errors are too large to treat it as reliable dose arithmetic, especially for norepinephrine. I’ll check whether the earlier atom embeddings retain a more precise rate signal. If they do, a small analytic channel may work; if the information is already lost there, this route cannot support a trustworthy repair.

```python
import dis
import numpy as np

def compact115(f, max_chars=2300):
    lines = []
    for ins in dis.Bytecode(f):
        if ins.opname in {"RESUME", "CACHE", "NOP", "EXTENDED_ARG"}:
            continue
        lines.append(f"{ins.opname} {ins.argrepr}")
    return "\n".join(lines)[:max_chars]

print("=== ACTION SET ENCODING ===")
print(compact115(action_hidden105, 2900))
print("\n=== ATTENTION IMPLEMENTATION ===")
print(compact115(wm_attention, 2900))
print("\n=== ACTION-AUDIT OBJECTS ===")
for n, v in list(globals().items()):
    if n.startswith("_") or not n.endswith("110"):
        continue
    if isinstance(v, dict):
        print(n, repr(v)[:800])
    elif isinstance(v, (float, int, str)):
        print(n, repr(v))
    elif hasattr(v, "coef_"):
        print(n, type(v).__name__, "coef shape", np.shape(v.coef_))
print("\ntracked / temporary:", tracked105, temporary105)
print("\n=== VOCAB NUMERIC NORMALISATION ===")
print("vocab keys:", list(VOCAB))
for k, v in VOCAB.items():
    if any(s in k.lower() for s in ["norm", "numeric", "config"]):
        print(k, repr(v)[:1300])
print("\n=== ATOM REGRESSION TEST ===")
from sklearn.linear_model import Ridge
for d, drug in enumerate(["insulin", "norepinephrine"]):
    a = np.asarray(atomtrain110[d], dtype=np.float64)
    b = np.asarray(atomtest110[d], dtype=np.float64)
    y = np.asarray(targettrain110[d], dtype=np.float64)
    z = np.asarray(targettest110[d], dtype=np.float64) if "targettest110" in globals() else None
    print(drug, "input", a.shape, "target", y.shape, "target first", y[:3])
    if z is not None:
        r = Ridge(alpha=1e-6).fit(a, y)
        err = r.predict(b) - z
        print("held-out RMSE", np.sqrt(np.mean(err**2, axis=0)).round(7),
              "max abs", np.max(np.abs(err), axis=0).round(5))
```
