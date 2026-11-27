The last result shows that a global glucose intercept has little consistent error to correct. It also identifies two spare neurons with small factual contributions. I’ll screen a blend of the gated and ungated insulin channels using those neurons. I’ll pursue it only if the screen suggests better dose response with a small NLL cost, then verify that on the GPU.

```python
from scipy.special import softmax

blend_base268 = {n: np.asarray(v, dtype=np.float32).copy()
                 for n, v in validated_gated8_checkpoint.items()}
blend_ungated268 = {n: np.asarray(v, dtype=np.float32).copy()
                    for n, v in stronger6_checkpoint.items()}
blend_nodes268 = tuple(calib_spare267[:2])
blend_oldnodes268 = tuple(map(int, sc_nodes))
blend_biascomp268 = (
    calib_w2orig267[:, list(blend_nodes268)]
    @ calib_hmean267[list(blend_nodes268)]
).astype(np.float32)

def make_blend268(a, b):
    out = {n: v.copy() for n, v in blend_base268.items()}
    out[sc_names[2]][:, list(blend_oldnodes268)] *= np.float32(a)
    if b != 0:
        for j, src in zip(blend_nodes268, blend_oldnodes268):
            out[sc_names[0]][j] = blend_ungated268[sc_names[0]][src]
            out[sc_names[1]][j] = blend_ungated268[sc_names[1]][src]
            out[sc_names[2]][:, j] = (
                np.float32(b) * blend_ungated268[sc_names[2]][:, src])
        out[sc_names[3]] += blend_biascomp268
    return out

def blend_parts268(c):
    x = np.asarray(c["x"], dtype=np.float32)
    def contribution(cp, nodes):
        h = F.gelu(torch.from_numpy(
            x @ cp[sc_names[0]][list(nodes)].T
            + cp[sc_names[1]][list(nodes)])).numpy()
        return h @ cp[sc_names[2]][:, list(nodes)].T
    return {
        "base": np.asarray(c["logits"], dtype=np.float32),
        "gated": contribution(blend_base268, blend_oldnodes268),
        "ungated": contribution(blend_ungated268, blend_oldnodes268),
        "removed": contribution(
            {n: np.asarray(v, dtype=np.float32) for n, v in sc_original.items()},
            blend_nodes268),
        "cache": c
    }

def blend_eval268(parts, a, b):
    logits = parts["base"] + a * parts["gated"]
    if b != 0:
        logits = (logits + b * parts["ungated"]
                  - parts["removed"] + blend_biascomp268)
    p = softmax(logits, axis=-1).mean(axis=0)
    c = parts["cache"]
    nll = -np.log(np.maximum(p[np.arange(len(p)),
                              np.asarray(c["bins"], dtype=int)], 1e-30))
    means = np.sum(p*np.asarray(c["centres"]), axis=-1)
    finger = np.asarray(c["meta"]["event"] == "glucose_finger")
    return float(nll.mean()), float(nll[finger].mean()), float(means[finger].mean())

blend_factparts268 = {k: blend_parts268(v) for k, v in sc_fact.items()}
blend_synparts268 = {k: blend_parts268(sc_syn[k]) for k in (0.5, 1.0, 2.0)}
blend_grid268 = [(1., 0.), (.85, .35), (.9, .25), (.8, .4),
                 (.75, .45), (.9, .35), (1., .1)]
blend_rows268 = []
for a, b in blend_grid268:
    row = {"gated_fraction": a, "ungated_fraction": b}
    for label, parts in blend_factparts268.items():
        nll, fnll, _ = blend_eval268(parts, a, b)
        base_nll, base_fnll, _ = blend_eval268(parts, 1., 0.)
        row[f"{label}_delta_vs_gated8"] = nll-base_nll
        row[f"{label}_finger_delta"] = fnll-base_fnll
    syn = {m: blend_eval268(parts, a, b)[2]
           for m, parts in blend_synparts268.items()}
    row["local_x2"] = syn[2.]-syn[1.]
    row["local_half"] = syn[.5]-syn[1.]
    blend_rows268.append(row)
print("New hidden neurons:", blend_nodes268)
print(pd.DataFrame(blend_rows268).round(6).to_string(index=False))
print("Current weights unchanged:",
      all(np.array_equal(np.asarray(load(n)), v)
          for n, v in blend_base268.items()))
with open("/work/notes.md", "a") as f:
    f.write("\n## Recovery and blend screening; Python 268, forwards 32/50\n"
            "- Verified current weights equal validated gated8.\n"
            "- GPU bin-centre errors are small and heterogeneous; "
            "a global intercept is not justified.\n"
            "- Screening mixtures with the validated ungated insulin channel "
            "using two additional low-contribution neurons.\n"
            "- CPU screens are design evidence only; no weights changed.\n")
```
