The screening stopped before changing any weights. The cache stores latent samples first, and my reconstruction also exceeded a tight numerical tolerance. I’ll correct the layout and compute only the logit changes. Agreement with the existing evaluator will validate the screen; it will not, by itself, validate dose rollouts.

```python
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

def n262_logits(weights, cache):
    def t(x):
        return torch.as_tensor(x, dtype=torch.float32)
    w0 = t(weights["attr_heads.value_bin.0.weight"])
    b0 = t(weights["attr_heads.value_bin.0.bias"])
    ow0 = t(n262_orig["attr_heads.value_bin.0.weight"])
    ob0 = t(n262_orig["attr_heads.value_bin.0.bias"])
    changed = ((w0-ow0).abs().amax(dim=1)>1e-7) | ((b0-ob0).abs()>1e-7)
    h0 = t(cache["H0"])
    hnew = h0.clone()
    hnew[...,changed] = F.gelu(F.linear(t(cache["X"]),w0[changed],b0[changed]))
    w2 = t(weights["attr_heads.value_bin.2.weight"])
    ow2 = t(n262_orig["attr_heads.value_bin.2.weight"])
    db2 = t(weights["attr_heads.value_bin.2.bias"])-t(n262_orig["attr_heads.value_bin.2.bias"])
    return (t(cache["L0"]) + F.linear(h0,w2-ow2)
            + F.linear(hnew-h0,w2) + db2)

def n262_table(weights, shrink=80):
    global r252_dmean
    raw = n262_logits(weights,c210_train)
    original = torch.as_tensor(c210_train["L0"],dtype=torch.float32)
    delta = (raw-original).numpy()
    assert delta.shape[1] == len(c210_train["meta"])
    prior = r252_dmean
    try:
        r252_dmean = delta.mean(axis=0)
        table = {k:np.asarray(v).copy()
                 for k,v in r252_fit_table(shrink).items()}
    finally:
        r252_dmean = prior
    return table

n263_reference = r246_candidate(.94,.50)
n263_ours = r252_nll(n262_logits(n263_reference,c210_train),c210_train)
n263_known = r246_loss(n263_reference,c210_train)
print("reconstruction NLL error: max",float(np.max(np.abs(n263_ours-n263_known))),
      "mean",float(np.mean(np.abs(n263_ours-n263_known))))
assert np.max(np.abs(n263_ours-n263_known)) < 2e-4
assert np.mean(np.abs(n263_ours-n263_known)) < 1e-5

n262_results,n262_candidates = [],{}
for ia,na in [(.94,.25),(.94,.30),(.94,.35),(.94,.40),
              (.90,.35),(.90,.40),(.86,.40),(.86,.45),(.94,.50)]:
    w = r246_candidate(ia,na)
    table = n262_table(w,80)
    rawtrain,rawtune = n262_logits(w,c210_train),n262_logits(w,c210_tune)
    for scale in [0.,2.,4.]:
        tr = r252_nll(rawtrain,c210_train,
                     -scale*r252_table_rows(table,c210_train))-r252_base["train"]
        tu = r252_nll(rawtune,c210_tune,
                     -scale*r252_table_rows(table,c210_tune))-r252_base["tune"]
        n262_results.append({
            "ins":ia,"ne":na,"cancel_scale":scale,
            "train_fit":float(tr[r252_fit].mean()),
            "train_validation":float(tr[r252_val].mean()),
            "tuning100":float(tu.mean())
        })
        n262_candidates[(ia,na,scale)] = (w,table)
n262_screen = pd.DataFrame(n262_results)
print(n262_screen.to_string(index=False,float_format=lambda x:f"{x:.6f}"))

with open("/work/notes.md","a") as f:
    f.write("\n## Runs 262–263: corrected amplitude screen\n"
            "Run262 stopped before screening or installing weights. Corrected the "
            "cache layout (32 latent samples, readings, channels) and delta-logit "
            "arithmetic; checked against the existing evaluator. No weights changed.\n")
    f.write(n262_screen.to_string(index=False)+"\n")
```
