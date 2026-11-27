The insulin edit uses two replacement neurons; the NE edit adds two more. The combined candidates’ factual losses remain too high on the second cohort, so I’ll check their actual rollout gains and recover the probe’s coordinate convention. This will show whether a simpler amplitude reduction is sufficient or whether the NE direction itself needs replacing.

```python
print("=== HISTORY AVAILABILITY ===")
print({k: k in globals() for k in ["In", "_ih", "get_ipython", "history", "runs"]})
print("=== NE DIRECTION RELATIONSHIPS ===")
for name in ["r177_beta", "r177_v", "r177_probe_direction",
             "r225_reference", "r225_d", "r223_ne_v", "r223_ne_common_v"]:
    a = r225_np(globals()[name])
    print(name, "norm", np.linalg.norm(a),
          "dot r225_d", float(a @ r225_d))
P229, sd229 = r177_projection, r177_sd
for label, v in [
    ("P(beta/sd)", P229 @ (r177_beta / sd229)),
    ("P(beta*sd)", P229 @ (r177_beta * sd229)),
    ("(P beta)/sd", (P229 @ r177_beta) / sd229),
    ("(P beta)*sd", (P229 @ r177_beta) * sd229),
    ("P beta", P229 @ r177_beta)
]:
    print(label, "error_vs_v", float(np.max(np.abs(v-r177_v))),
          "dot_raw_d", float(v @ r225_d))
print("=== PROBE FEATURE CONTENT ===")
for name in ["r137_probe_features", "r138_features"]:
    d = globals()[name]
    for key in list(d)[:1]:
        val = d[key]
        print(name, key, type(val).__name__)
        if isinstance(val, dict):
            print([(k, getattr(v, "shape", type(v).__name__)) for k,v in val.items()])
        elif isinstance(val, (tuple,list)):
            print([(type(v).__name__, getattr(v, "shape", "")) for v in val])
print("=== FORWARD FILES 20–28 ===")
for k in range(20,29):
    folder = Path(f"/work/fwd_{k}")
    print(k, [p.name for p in folder.iterdir()])
    for p in folder.glob("dose_edit_*.csv"):
        d = pd.read_csv(p)
        print(p.name, d.columns.tolist(), d.head(4).to_dict("records"))
    for p in folder.glob("factual_*.csv"):
        d = pd.read_csv(p)
        cols = [c for c in d.columns if "nll" in c.lower()]
        print(p.name, len(d), {c: float(d[c].mean()) for c in cols})
print("=== SELECTED NOTEBOOK SECTIONS ===")
for section in sections228:
    if (any(s in section[:90] for s in ["177", "176", "Forward 24", "Forward 25", "Forward 26"])
        or section.startswith("Run 8:") or section.startswith("Run 9:")):
        print(section[:1400])
```
