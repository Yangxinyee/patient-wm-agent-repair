The provisional targets are insulin about −14.2/+7.1 and norepinephrine +4.7/−2.3. The last validated insulin response was already close, so I’ll first screen reductions in norepinephrine strength, refitting factual cancellation only on training records. This would help if it lowers NLL while leaving insulin near its target; rollout validation is still necessary.

```python
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

torch.set_num_threads(8)
n262_orig = {k: np.asarray(load_original(k), dtype=np.float32)
             for k in r246_safe_np}

def n262_logits(weights, cache):
    def t(x):
        return torch.as_tensor(x, dtype=torch.float32)
    h = F.gelu(F.linear(t(cache["X"]),
                        t(weights["attr_heads.value_bin.0.weight"]),
                        t(weights["attr_heads.value_bin.0.bias"])))
    new = F.linear(h, t(weights["attr_heads.value_bin.2.weight"]),
                   t(weights["attr_heads.value_bin.2.bias"]))
    old = F.linear(t(cache["H0"]),
                   t(n262_orig["attr_heads.value_bin.2.weight"]),
                   t(n262_orig["attr_heads.value_bin.2.bias"]))
    return t(cache["L0"]) + new - old

def n262_table(weights, shrink=80):
    global r252_dmean
    raw = n262_logits(weights, c210_train)
    original = torch.as_tensor(c210_train["L0"], dtype=torch.float32)
    delta = (raw-original).numpy()
    assert delta.shape[0] == len(c210_train["meta"])
    sample_axes = tuple(range(1, delta.ndim-1))
    dmean = delta.mean(axis=sample_axes) if sample_axes else delta
    prior = r252_dmean
    try:
        r252_dmean = dmean
        table = {k: np.asarray(v).copy()
                 for k,v in r252_fit_table(shrink).items()}
    finally:
        r252_dmean = prior
    return table

print("=== CACHE SHAPES ===")
for nm,c in [("train",c210_train),("tuning100",c210_tune)]:
    print(nm, {k:tuple(c[k].shape) for k in ["X","H0","L0"]})

n262_reference = r246_candidate(.94, .50)
n262_reference_loss = r252_nll(n262_logits(n262_reference,c210_train),c210_train)
print("head reconstruction max NLL error:",
      float(np.max(np.abs(n262_reference_loss-r246_loss(n262_reference,c210_train)))))
assert np.max(np.abs(n262_reference_loss-r246_loss(n262_reference,c210_train))) < 2e-5

n262_results, n262_candidates = [], {}
for ia,na in [(.94,.25),(.94,.30),(.94,.35),(.94,.40),
              (.90,.35),(.90,.40),(.86,.40),(.86,.45),(.94,.50)]:
    w = r246_candidate(ia,na)
    table = n262_table(w,80)
    rawtrain, rawtune = n262_logits(w,c210_train), n262_logits(w,c210_tune)
    for scale in [0.,2.,4.]:
        trcorr = -scale*r252_table_rows(table,c210_train)
        tucorr = -scale*r252_table_rows(table,c210_tune)
        tr = r252_nll(rawtrain,c210_train,trcorr)-r252_base["train"]
        tu = r252_nll(rawtune,c210_tune,tucorr)-r252_base["tune"]
        n262_results.append({
            "ins":ia,"ne":na,"cancel_scale":scale,
            "train_fit":float(tr[r252_fit].mean()),
            "train_validation":float(tr[r252_val].mean()),
            "tuning100":float(tu.mean())
        })
        n262_candidates[(ia,na,scale)] = (w,table)
n262_screen = pd.DataFrame(n262_results)
print("\n=== AMPLITUDE / TRAINING-ONLY CANCELLATION SCREEN ===")
print(n262_screen.to_string(index=False,float_format=lambda x:f"{x:.6f}"))

with open("/work/notes.md","a") as f:
    f.write("\n## Run 262: reduced-amplitude factual screen\n"
            "Reconstructed edited head logits against the cached evaluator. "
            "Screened insulin/NE amplitudes and training-only last-bin cancellation; "
            "no weights installed. Provisional record-derived targets remain "
            "insulin approximately -14.18/+7.09 and NE +4.68/-2.34.\n")
    f.write(n262_screen.to_string(index=False)+"\n")
```
