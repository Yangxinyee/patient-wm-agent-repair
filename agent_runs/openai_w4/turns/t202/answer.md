The channel uses two replaced posterior units, but the previous output cut off the gating details. I’ll isolate those coefficients and their training targets. The intended design is a dose deviation that is active only on drug bundles; if the coefficients or held-out behavior do not support that, I’ll use a different encoding rather than assume it works.

```python
import io, dis, inspect
from pathlib import Path

buf177 = io.StringIO()
dis.dis(rp_mlp, file=buf177, depth=0)
print("MLP", buf177.getvalue()[:2600])

def arr177(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

w0177 = arr177(load("z_post.0.weight"))
b0177 = arr177(load("z_post.0.bias"))
w2177 = arr177(load("z_post.2.weight"))
for j177 in [96, 184]:
    print("HIDDEN", j177, "bias", float(b0177[j177]),
          "block norms", [float(np.linalg.norm(w0177[j177, s:s+192]))
                          for s in [0, 192, 384]],
          "output norm", float(np.linalg.norm(w2177[:, j177])),
          "axis loading", float(arr177(latent_axis104) @ w2177[:32, j177]))
print("ROW SUM/DIFF",
      float(np.linalg.norm(w0177[96] + w0177[184])),
      float(np.linalg.norm(w0177[96] - w0177[184])))

print("\nPOTENTIAL FIT OBJECTS")
names177 = sorted(k for k in list(globals())
                  if any(s in k.lower() for s in
                         ["posterior", "gated", "innov", "coef", "theta",
                          "dose_predict", "policy", "raw_features", "gate_vector"]))
for k177 in names177[:90]:
    v177 = globals()[k177]
    if callable(v177):
        print(k177, inspect.signature(v177), v177.__code__.co_names[:14])
    elif isinstance(v177, dict):
        print(k177, "dict", list(v177)[:12])
    elif hasattr(v177, "shape"):
        print(k177, tuple(v177.shape))
    elif isinstance(v177, (str, int, float, list, tuple)):
        print(k177, str(v177)[:100])

print("\nREFERENCE ARRAY SHAPES")
for k177 in ["U65", "E65", "indices102", "pan", "D", "rates", "dose_probe104"]:
    if k177 in globals():
        v177 = globals()[k177]
        print(k177, type(v177).__name__, getattr(v177, "shape", None))
        if hasattr(v177, "columns"):
            print("columns", list(v177.columns)[:30], "index", v177.index.names,
                  "head", v177.head(2).to_string()[:800])
for drug177, dd177 in dml_residuals.items():
    print(drug177, {k: getattr(v, "shape", type(v).__name__)
                    for k, v in dd177.items()})

print("\nEARLIER GATING NOTES")
for marker177 in ["## Run 89", "## Run 90", "## Run 91",
                  "## Run 92", "## Run 93", "## Run 94", "## Run 95",
                  "## Run 96", "## Run 97", "## Run 98", "## Run 99"]:
    p177 = notes174.find(marker177)
    if p177 >= 0:
        e177 = notes174.find("\n## ", p177 + 4)
        print(notes174[p177:e177 if e177 >= 0 else p177 + 700][:700])

with Path("/work/notes.md").open("a") as f177:
    f177.write("\n\n## Run 177\n"
               "Recovered compact gating/MLP diagnostics and record-target interfaces after "
               "earlier output truncation. No weights changed. NE encoding will be checked "
               "against actual recorded doses and held-out bundles rather than medical priors.\n")
```